import os
import torch
import torch.nn as nn
from models.model_deblur import VGG_Deblur
from dataset import MyDataset
from utils.losses import CharbonnierLoss
from test_model.test_model import test_state
from utils.Flowfunction import flow_color
from models.pwc_net import PWCNET
from utils.warp import get_backwarp
from models.kpn import Reblur_Model
EPS=1e-12
norm_val=(2**8)-1
src_mean=0

class Solver:
    def __init__(self,args):
        self.args = args
        self.start_epoch=0
        self.global_step = 0

    def prepare_data(self,train_path):
        im_src=[]
        im_gt=[]
        
        path_gt= train_path+"gt/"
        path_src = train_path+"blur/"

        for _,_,fnames in sorted(os.walk(path_gt)):
            for fname in fnames:                
                im_src.append(path_src+fname)
                im_gt.append(path_gt+fname)
        return im_gt,im_src
    
    def mask_gene(self,flow,sigma):
        flow_gray = flow_color(flow)        
        base = torch.ones((flow_gray.shape)).cuda()
        flow_gray = torch.Tensor(flow_gray).cuda()
        for i in range(flow_gray.shape[0]):
            base[i,:] *=torch.mean(flow_gray,[1,2,3])[i]
            
        mask = torch.ones((flow_gray.shape)).cuda()
        mask[flow_gray<(1-sigma)*base] = 0
        mask[flow_gray>(1+sigma)*base] = 0
        
        return mask
    
    def train_model(self):
        self.deblur_model = VGG_Deblur()
        self.deblur_model = nn.DataParallel(self.deblur_model)
        self.deblur_model.cuda()
        
        self.pwcnet = PWCNET()
        self.pwcnet.cuda()
        self.pwcnet = nn.DataParallel(self.pwcnet)
        
        self.reblur_model = Reblur_Model()
        if self.args.kpn == 'kpn-onebranch':
            from models.kpn_onebranch import Reblur_Model_1
            print('using kpn-onebranch')
            self.reblur_model = Reblur_Model_1()
        self.reblur_model = nn.DataParallel(self.reblur_model)
        self.reblur_model.cuda()      
        
        self.lr = self.args.lr
        param = list(self.deblur_model.parameters()) + list(self.reblur_model.parameters()) 
        self.deblur_opt = torch.optim.Adam(param,lr=self.lr,)
        
        lr_=[]
        lr_.append(self.lr) #initial learning rate
        for i in range(int(self.args.num_epochs/self.args.lr_decay)):
            lr_.append(lr_[i]*0.5)

        #optional resume from a checkpoit
        if self.args.resume_file:
            if os.path.isfile(self.args.resume_file):
                print("loading checkpoint'{}'".format(self.args.resume_file))
                checkpoint = torch.load(self.args.resume_file)
                self.start_epoch = checkpoint['epoch']
                self.global_step = checkpoint['global_step']
                self.deblur_model.load_state_dict(checkpoint['G_state_dict'])
                self.reblur_model.load_state_dict(checkpoint['Reblur_state_dict'])
                self.deblur_opt.load_state_dict(checkpoint['G_opt'])
                
                del(checkpoint)
                print("'{}' loaded".format(self.args.resume_file,self.args.start_epoch))
            else:
                print("no checkpoint found at '{}'".format(self.args.resume_file))
                return 1

        torch.backends.cudnn.benchmark = True

        im_gt,im_src=self.prepare_data(self.args.data_path_single)
        train_dpdd_dataset = MyDataset(im_gt,im_src)
        train_dpdd_loader = torch.utils.data.DataLoader(dataset=train_dpdd_dataset,batch_size=self.args.batch_size,shuffle=True,
                    num_workers=self.args.load_workers)

        # train the model
        best_psnr = best_ssim = 0
        
        for epoch in range(self.start_epoch, self.args.num_epochs):
            self.lr = lr_[int(epoch/self.args.lr_decay)]
            for param_group in self.deblur_opt.param_groups:
                param_group['lr'] = self.lr
                
            G_loss_avg = self.train_epoch(train_dpdd_loader,epoch)
            
            if epoch % self.args.save_model_freq == 0 or epoch%self.args.test_model_freq == 0:
                state = {
                    'epoch': epoch + 1,
                    'global_step': self.global_step,
                    'G_state_dict': self.deblur_model.state_dict(),
                    'Reblur_state_dict': self.reblur_model.state_dict(),
                    'G_opt': self.deblur_opt.state_dict(),
                }

                ssim,psnr,mse = test_state(self.args,state['G_state_dict'])

                if not os.path.exists(self.args.weight_save_path):
                    os.mkdir(self.args.weight_save_path)
                
                if epoch % self.args.save_model_freq == 0:
                    torch.save(state, self.args.weight_save_path+'/epoch_{:0>3}_G_{:.3f}_P_{:.3f}.pth'.format(epoch,G_loss_avg,psnr))
                    
                if psnr > best_psnr or ssim > best_ssim:
                    print('Saving checkpoint, psnr: {} ssim: {}'.format(psnr,ssim))
                    if psnr > best_psnr:
                        best_psnr = psnr
                        torch.save(state, self.args.weight_save_path+'/epoch_{:0>3}_G_{:.3f}_P_{:.3f}.pth'.format(epoch,G_loss_avg,psnr))
                    else:
                        best_ssim = ssim
                        torch.save(state, self.args.weight_save_path+'/epoch_{:0>3}_G_{:.3f}_S_{:.3f}.pth'.format(epoch,G_loss_avg,ssim))


    def train_epoch(self,train_dpdd_loader,epoch):
        self.deblur_model.train()
        self.reblur_model.train()
        loss_fn = CharbonnierLoss()                    
        G_loss_sum=0
        for index, (im_gt,im_b) in enumerate(train_dpdd_loader):
               
            im_gt = im_gt.cuda()
            im_b = im_b.cuda()
            output = self.deblur_model(im_b)
            
            if epoch < 15:
                wrap_gt, flow = get_backwarp(im_b, im_gt, self.pwcnet)
                char_loss = loss_fn(output, wrap_gt)
                loss = char_loss
                
                if index % self.args.print_freq == 0:
                    print('loss: {0}\tstep: {1}\tepoch: {2}\tlr: {3}'.format(loss.item(),index,epoch,self.lr))  

            else:
                wrap_gt, flow = get_backwarp(output, im_gt, self.pwcnet)
                warp_out, flow_ = get_backwarp(im_gt, output, self.pwcnet)

                with torch.no_grad():
                    mask = self.mask_gene(flow,0.35)
                    mask_ = self.mask_gene(flow_,0.35)

                char_loss = loss_fn(mask*output, mask*wrap_gt) + loss_fn(mask_*im_gt, mask_*warp_out)
                reblur_out = self.reblur_model(output, im_b)
                torch.cuda.empty_cache()
                reblur_loss = loss_fn(reblur_out, im_b)
                
                loss = char_loss + reblur_loss*0.5 
                
                if index % self.args.print_freq == 0:
                    print('loss: {0}\tstep: {1}\tepoch: {2}\tlr: {3}'.format(loss.item(),index,epoch,self.lr))

            self.deblur_opt.zero_grad()
            
            G_loss_sum += loss.item()
            loss = loss.cuda()
            loss.backward()
            
            self.deblur_opt.step()
            torch.cuda.empty_cache()
            

        self.global_step+=1
        return G_loss_sum/len(train_dpdd_loader)
