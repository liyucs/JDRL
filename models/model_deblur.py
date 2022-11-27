import torch
import torch.nn as nn
from utils.padding import Conv2d


#########UPP funcs#####################################################
class Encoder_block(nn.Module):
    def __init__(self):
        super(Encoder_block, self).__init__()
        self.block1 = nn.Sequential(
            Conv2d(3, 64, 3),
            nn.ReLU(),
            Conv2d(64, 64, 3),
            nn.ReLU())
        self.max1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.block2 = nn.Sequential(
            Conv2d(64, 128, 3),
            nn.ReLU(),
            Conv2d(128, 128, 3),
            nn.ReLU())
        self.max2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.block3 = nn.Sequential(
            Conv2d(128, 256, 3),
            nn.ReLU(),
            Conv2d(256, 256, 3),
            nn.ReLU())
        self.max3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.block4 = nn.Sequential(
            Conv2d(256, 512, 3),
            nn.ReLU(),
            Conv2d(512, 512, 3),
            nn.ReLU())
        self.drop4 = nn.Dropout(p=0.4)
        self.max4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.block5 = nn.Sequential(
            Conv2d(512, 1024, 3),
            nn.ReLU(),
            Conv2d(1024, 1024, 3),
            nn.ReLU())
        self.drop5 = nn.Dropout(p=0.4)    
    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.block2(self.max1(x1))
        x3 = self.block3(self.max2(x2))
        x4 = self.drop4(self.block4(self.max3(x3)))
        torch.cuda.empty_cache()
        x5 = self.drop5(self.block5(self.max4(x4)))
        torch.cuda.empty_cache()
        return x1, x2, x3, x4, x5
        
    
class VGG_Deblur(nn.Module):
    def __init__(self):
        super(VGG_Deblur, self).__init__()
        self.encoder = Encoder_block()
        self.up6 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            Conv2d(1024,512,2))
        self.block6 = nn.Sequential(
            Conv2d(1024, 512, 3),
            nn.ReLU(inplace=True),
            Conv2d(512, 512, 3),
            nn.ReLU(inplace=True))
        self.up7 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            Conv2d(512,256,2))
        self.block7 = nn.Sequential(
            Conv2d(512, 256, 3),
            nn.ReLU(inplace=True),
            Conv2d(256, 256, 3),
            nn.ReLU(inplace=True))        
        self.up8 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            Conv2d(256,128,2))
        self.block8 = nn.Sequential(
            Conv2d(256, 128, 3),
            nn.ReLU(inplace=True),
            Conv2d(128, 128, 3),
            nn.ReLU(inplace=True))
        self.up9 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            Conv2d(128,64,2))
        self.block9 = nn.Sequential(
            Conv2d(128, 64, 3),
            nn.ReLU(inplace=True),
            Conv2d(64, 64, 3),
            nn.ReLU(inplace=True))          
        self.out = nn.Sequential(
            Conv2d(64, 3, 3))
#            nn.ReLU(inplace=True))
        
        self.__init_weight()
        
    def forward(self, im_b):
        x1, x2, x3, x4, x5 = self.encoder(im_b)
        x6_up = self.up6(x5)
        x6 = self.block6(torch.cat([x6_up,x4], dim=1))
        x7_up = self.up7(x6)
        x7 = self.block7(torch.cat([x7_up,x3], dim=1))
        x8_up = self.up8(x7)
        x8 = self.block8(torch.cat([x8_up,x2], dim=1))
        x9_up = self.up9(x8)
        x9 = self.block9(torch.cat([x9_up,x1], dim=1))
        out = self.out(x9)
        torch.cuda.empty_cache()
        
        return torch.sigmoid(out)

    
    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, Conv2d):
                nn.init.kaiming_normal_(m.weight,mode = 'fan_out',nonlinearity='relu')
                




                
        
              
                
                
                
                
                
                
                
                
                
                
                
