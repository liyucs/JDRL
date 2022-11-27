import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.arch_util as arch_util
import functools


def flipcat(x, dim):
    return torch.cat([x,x.flip(dim).index_select(dim,torch.arange(1,x.size(dim)).cuda())],dim)
    
class KPN(nn.Module):
    def __init__(self, nf=64):
        super(KPN,self).__init__()
        
        self.conv_first = nn.Conv2d(6, nf, 3, 1, 1, bias=True)
        basic_block = functools.partial(arch_util.ResidualBlock_noBN, nf=nf)
        self.block1 = basic_block()
        self.block2 = basic_block()
        self.block3 = basic_block()
        self.out = nn.Conv2d(64, 35, 3, 1, 1, bias=True)
        self.kernel_pred = KernelConv()
        
    def forward(self, data_with_est, data):
        x = self.conv_first(data_with_est)
        x = self.block3(self.block2(self.block1(x)))
        core = self.out(x)
        
        return self.kernel_pred(data, core)


class KernelConv(nn.Module):
    """
    the class of computing prediction
    """
    def __init__(self):
        super(KernelConv, self).__init__()
    
    def _list_core(self, core_list, batch_size, N, color, height, width):
        """
        convert the sep_conv core to conv2d core
        2p --> p^2
        :param core: shape: batch*(N*2*K)*height*width
        :return:
        """
        core_out = {}
    
        for i in range(len(core_list)):
            core = core_list[i]
            core = torch.abs(core)
            wide = core.shape[1]
            final_wide = (core.shape[1]-1)*2+1
            kernel = torch.zeros((batch_size,wide,wide,color,height, width)).cuda()
            mid = torch.Tensor([core.shape[1]-1]).cuda()
            for i in range(wide):
                for j in range(wide):
                    distance = torch.sqrt((i-mid)**2 + (j-mid)**2)
                    low = torch.floor(distance)
                    high = torch.ceil(distance)
                    if distance > mid:
                        kernel[:,i,j,0,:,:] = 0
                    elif low == high:
                        kernel[:,i,j,0,:,:] = core[:,int(low),:,:]
                    else:
                        y0 = core[:,int(low),:,:]
                        y1 = core[:,int(low)+1,:,:]
                        kernel[:,i,j,0,:,:] = (distance-low)*y1 + (high-distance)*y0
                    
            kernel = flipcat(flipcat(kernel,1), 2)   
            core_ori = kernel.view(batch_size, N, final_wide * final_wide, color, height, width)
            core_out[final_wide] = F.softmax(core_ori,dim=2)

        return core_out

    def forward(self, frames, core):
        """
        compute the pred image according to core and frames
        :param frames: [batch_size, N, 3, height, width]
        :param core: [batch_size, N, dict(kernel), 3, height, width]
        :return:
        """
        pred_img = [frames]
        batch_size, N, height, width = frames.size()
        color = 1
        frames = frames.view(batch_size, N, color, height, width)
        
        section = [2,3,4,5,6,7,8]
        core_list = []
        core_list = torch.split(core, section, dim=1)

        core_out = self._list_core(core_list, batch_size, 1, color, height, width)                
        kernel_list = [3,5,7,9,11,13,15]
        
                
        for index, K in enumerate(kernel_list):
            img_stack = []
            frame_pad = F.pad(frames, [K // 2, K // 2, K // 2, K // 2])
            for i in range(K):
                for j in range(K):
                    img_stack.append(frame_pad[..., i:i + height, j:j + width])
            img_stack = torch.stack(img_stack, dim=2)
            pred = torch.sum(core_out[K].mul(img_stack), dim=2, keepdim=False)
            if batch_size == 1:
                pred = pred.squeeze().unsqueeze(0)
            else:
                pred = pred.squeeze()
            pred_img.append(pred)

        return pred_img


class KERNEL_MAP(nn.Module):
    def __init__(self, nf=64):
        super(KERNEL_MAP,self).__init__()
        
        self.conv_first = nn.Conv2d(6, nf, 3, 1, 1, bias=True)
        basic_block = functools.partial(arch_util.ResidualBlock_noBN, nf=nf)
        self.block1 = basic_block()
        self.block2 = basic_block()
        self.block3 = basic_block()
        self.out = nn.Conv2d(64, 8, 3, 1, 1, bias=True)
        
    def forward(self, x):
        x = self.conv_first(x)
        x = self.block3(self.block2(self.block1(x)))
        kernel_map = self.out(x)
        kernel_map = F.softmax(kernel_map, dim=1)
        return kernel_map
    
class Reblur_Model(nn.Module):
    def __init__(self):
        super(Reblur_Model, self).__init__()
        self.kernel_map_gene = KERNEL_MAP()
        self.kpn = KPN()
        self.apply(self._init_weights)
    
    @staticmethod     
    def _init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight.data)
            nn.init.constant_(m.bias.data, 0.0)
    
    def forward(self,im_s,im_b):
        est_input = torch.cat([im_s,im_b], dim=1)
        pred_img = self.kpn(est_input, im_s)
        kernel_map = self.kernel_map_gene(est_input)
        
        map0,map1,map2,map3,map4,map5,map6,map7 = torch.chunk(kernel_map, 8, dim=1)
        map_list = [map0,map1,map2,map3,map4,map5,map6,map7]
        output = map0*pred_img[0]
        for n in range(1,8):
            output += (pred_img[n])*(map_list[n])
        
        return output
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    