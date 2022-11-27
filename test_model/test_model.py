import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
from models.model_deblur import VGG_Deblur
from skimage import img_as_ubyte
from models.MPRNet import MPRNet
import torch.nn as nn
from PIL import Image
import os
from utils.warp import get_backwarp
import torchvision.transforms.functional as TF
from models.pwc_net import PWCNET
import cv2
from utils import index

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

norm_val=(2**8)-1
src_mean = 0

def creat_list(path):
    gt_list = []
    im_list = []
    im_path = path + 'blur/'
    gt_path = path + 'gt/'

    for _,_,fnames in sorted(os.walk(gt_path)):
        for fname in fnames:
            gt_list.append(gt_path+fname)
            im_list.append(im_path+fname)

    return gt_list,im_list
to_tensor = transforms.ToTensor()            

class TestDataset(Dataset):
    def __init__(self,args,gt_list,im_list):
        self.gt_list = gt_list
        self.im_list = im_list
        self.args = args

    def __getitem__(self, index):
        if self.args.model == 'UNet':
            gt = cv2.imread(self.gt_list[index])     
            im = cv2.imread(self.im_list[index])
            
            gt = np.float32((gt-src_mean)/norm_val)
            im = np.float32((im-src_mean)/norm_val)
    
            gt=torch.from_numpy(gt).permute(2,0,1)
            im=torch.from_numpy(im).permute(2,0,1)
            
            c,h,w = gt.shape
            H = h//16*16
            W = w//16*16
            
            gt = gt[:,0:H,0:W]
            im = im[:,0:H,0:W]
            
            return gt, im
        
        if self.args.model == "MPRNet":
            inp = Image.open(self.im_list[index])
            tar = Image.open(self.gt_list[index])

            h,w = inp.size
            inp = TF.center_crop(inp, (w//8*8,h//8*8))
            tar = TF.center_crop(tar, (w//8*8,h//8*8))
                 
            inp = TF.to_tensor(inp)
            tar = TF.to_tensor(tar)
            return tar, inp
            
    
    def __len__(self):
        return len(self.gt_list)

def test_dataset(args,deblur_model,test_loader,save_path=None):
    ssim_sum = 0
    mse_sum = 0
    psnr_sum = 0
    
    for j, (gt, im) in enumerate(test_loader):
        gt = gt.cuda()
        im = im.cuda()

        with torch.no_grad():
            if args.model == 'UNet':
                output  = deblur_model(im)     
            if args.model == 'MPRNet':
                restored  = deblur_model(im)
                output = torch.clamp(restored[0],0,1)
            wrap_gt, flowl = get_backwarp(output, gt, pwcnet)
                
            gt = gt[0,...].permute(1,2,0).cpu().detach().numpy()
            output = output[0,...].permute(1,2,0).cpu().detach().numpy()
            wrap_gt = wrap_gt[0,...].permute(1,2,0).cpu().detach().numpy()
            im = im[0,...].permute(1,2,0).cpu().detach().numpy()
            
            if args.test_path == './SDD/test/':
                mse, psnr, ssim = index.MSE_PSNR_SSIM(wrap_gt.astype(np.float64), output.astype(np.float64))
            if args.test_path == './DPDD/test/':
                mse, psnr, ssim = index.MSE_PSNR_SSIM(gt.astype(np.float64), output.astype(np.float64))
                
            ssim_sum += ssim
            mse_sum += mse
            psnr_sum += psnr
            print('SSIM:',ssim,'PSNR:',psnr,'MSE:',mse)
 
            if save_path:
                if not os.path.exists(save_path):
                    os.mkdir(save_path) 
                if args.model == 'UNet':
                    cv2.imwrite("%s/%s.png"%(save_path,j),np.uint8(im*norm_val))
                    cv2.imwrite("%s/%s_wgt.png"%(save_path,j),np.uint8(wrap_gt*norm_val))
                    cv2.imwrite("%s/%s_gt.png"%(save_path,j),np.uint8(gt*norm_val))
                    cv2.imwrite("%s/%s_output.png"%(save_path,j),np.uint8(output*norm_val))
                if args.model == "MPRNet":
                    cv2.imwrite("%s/%s.png"%(save_path,j),cv2.cvtColor(np.uint8(im*norm_val), cv2.COLOR_RGB2BGR))
                    cv2.imwrite("%s/%s_wgt.png"%(save_path,j),cv2.cvtColor(np.uint8(wrap_gt*norm_val), cv2.COLOR_RGB2BGR)) 
                    cv2.imwrite("%s/%s_gt.png"%(save_path,j),cv2.cvtColor(np.uint8(gt*norm_val), cv2.COLOR_RGB2BGR)) 
                    cv2.imwrite("%s/%s_output.png"%(save_path,j),cv2.cvtColor(np.uint8(output*norm_val), cv2.COLOR_RGB2BGR)) 

    print(len(test_loader),'SSIM:',ssim_sum/len(test_loader),'PSNR:',psnr_sum/len(test_loader),'MSE:',mse_sum/len(test_loader))
    return len(test_loader),ssim_sum/len(test_loader),psnr_sum/len(test_loader),mse_sum/len(test_loader)

def add(num_,ssim_sum_,psnr_sum_,lmse_sum_,ncc_sum_,                                                                                                                                                                                                                                                                                                                                                                                                                                                        
        num_test,ssim_sum_test,psnr_sum_test,lmse_sum_test,ncc_sum_test):
    return num_+num_test,ssim_sum_+ssim_sum_test,psnr_sum_+psnr_sum_test,lmse_sum_+lmse_sum_test,ncc_sum_+ncc_sum_test


pwcnet = PWCNET()
pwcnet = nn.DataParallel(pwcnet)
pwcnet.cuda() 
pwcnet.eval()


def test_state(args,state_dict):
    
    if args.model == 'MPRNet':
        deblur_model = MPRNet()
    if args.model == 'UNet':
        deblur_model = VGG_Deblur()
        
    gt_list,im_list = creat_list(args.test_path)
    test_dpdd_dataset = TestDataset(args,gt_list,im_list)
    test_loader_dpdd = torch.utils.data.DataLoader(dataset=test_dpdd_dataset,batch_size=1,shuffle=False,num_workers=args.num_workers)
    
    deblur_model = nn.DataParallel(deblur_model)
    deblur_model.cuda()
    deblur_model.eval()
    deblur_model.load_state_dict(state_dict)
    del(state_dict) 
    save_path = None

    if args.save_result_path == True:
        save_path = './result_sdd' + args.model
        if args.test_path == './DPDD/test/':
            save_path = './result_dpdd' + args.model
    num, ssim_av,psnr_av,mse_av = test_dataset(args,deblur_model,test_loader_dpdd,save_path)

    return ssim_av,psnr_av,mse_av




    


