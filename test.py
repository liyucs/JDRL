import torch
from test_model.test_model import test_state
import matplotlib.pyplot as plt
import os

import argparse

parser = argparse.ArgumentParser('test')
parser.add_argument('--checkpoint_folder',default='',help="path to checkpoint folder, use when find best ckpt")
parser.add_argument('--test_path',default='./SDD/test/',help="path to test set")
parser.add_argument('--checkpoint_path',default="./checkpoint/mprnet-jdrl-sdd.pth",help="path to checkpoint")
parser.add_argument('--save_result_path',default=True,help="if save result")
parser.add_argument('--num_workers',default=4,help="num_workers")
parser.add_argument('--model',default='MPRNet',help="MPRNet or UNet")

args = parser.parse_args()


ckpt_list = []
ckpt_name_list = []
if args.checkpoint_folder:
    for _,_,ckfnames in sorted(os.walk(args.checkpoint_folder)):
        for ckfname in ckfnames:
            ckpt_path = args.checkpoint_folder + ckfname
            ckpt_list.append(ckpt_path)
            ckpt_name_list.append(ckfname[6:9])
if args.checkpoint_path:
    ckpt_list.append(args.checkpoint_path)
    ckpt_name_list.append('ckpt')
print(ckpt_list)

best_psnr = best_ssim = 0
second_psnr = second_ssim = 0
third_psnr = third_ssim = 0
name_psnr = 0
second_name = 0
psnr = []
ssim = []
mse = []    
mae = []

for i in range(len(ckpt_list)):
    
    ckpt_path_load  = ckpt_list[i]
    ckpt = torch.load(ckpt_path_load)
    ckpt_name = ckpt_name_list[i]
    print("loading checjpoint'{}'".format(ckpt_path_load))
    print(i)
    
    if args.model == 'UNet':
        state = ckpt['G_state_dict']
    if args.model == 'MPRNet':
        state = ckpt["state_dict"]
    ssim_av,psnr_av,mse_av = test_state(args,state)
    del(ckpt)

    psnr.append(psnr_av)
    ssim.append(ssim_av)

    if psnr_av > best_psnr:
        third_psnr = second_psnr
        third_name = second_name
        second_psnr = best_psnr
        second_name = name_psnr
        
        best_psnr = psnr_av
        name_psnr = ckpt_name
    if ssim_av > best_ssim:
        best_ssim = ssim_av
        name_ssim = ckpt_name

print('The best psnr:',best_psnr,name_psnr)
print('The second psnr:',second_psnr,second_name)
print('The third psnr:',third_psnr,third_name)
print('The best ssim:',best_ssim,name_ssim)

x,y=zip(*sorted(zip(ckpt_name_list,psnr)))
plt.plot(x,y)
plt.grid(linestyle='-.')
plt.savefig('result_all.jpg')
plt.show()

    
    


