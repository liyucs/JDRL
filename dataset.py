import cv2
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from utils.utils import syn
import torchvision.transforms as transforms  

to_tensor = transforms.ToTensor()    
norm_val=(2**8)-1
src_mean = 0

def Aug(img,mode):
    if mode == 0:
        out = img
    elif mode == 1:
        out = np.flipud(img)
    elif mode == 2:
        out = np.rot90(img)
    elif mode == 3:
        out = np.rot90(img)
        out = np.flipud(out)
    elif mode == 4:
        out = np.rot90(img, k=2)
    elif mode ==5:
        out = np.rot90(img, k=2)
        out = np.flipud(out)
    elif mode == 6:
        out = np.rot90(img, k=3)
    elif mode == 7:
        out = np.rot90(img, k=3)
        out = np.flipud(out)
        
    return out

def Crop_img(img,size):    
    crop_w = img.shape[0] - size
    crop_h = img.shape[1] - size
    if crop_w == 0:
        random_w = 0
    else:
        random_w = int(np.random.randint(0,crop_w)/2)
    if crop_h == 0:
        random_h = 0
    else:
        random_h = int(np.random.randint(0,crop_h)/2)
            
    return random_w,random_h
    

class MyDataset(Dataset):
    def __init__(self,im_gt,im_src,crop=Crop_img):
       
        self.im_gt_list=im_gt
        self.im_src_list = im_src
        self.crop = Crop_img

    def __getitem__(self, index):
        
        img_gt = cv2.imread(self.im_gt_list[index])
        while img_gt is None:
            img_gt = cv2.imread(self.im_gt_list[index])
        
        img_src = cv2.imread(self.im_src_list[index])
        while img_src is None:
            img_src = cv2.imread(self.im_src_list[index])  

        img_gt = np.float32((img_gt-src_mean)/norm_val)
        img_src = np.float32((img_src-src_mean)/norm_val)
                   
        random_w,random_h = self.crop(img_gt,256)
        img_gt = img_gt[random_w:random_w+256,random_h:random_h+256]
        img_src = img_src[random_w:random_w+256,random_h:random_h+256]
        
        img_gt=torch.from_numpy(img_gt).permute(2,0,1)
        img_src=torch.from_numpy(img_src).permute(2,0,1)
       
        return img_gt,img_src

    def __len__(self):
        return len(self.im_gt_list)
        
        
        
        
        
        
