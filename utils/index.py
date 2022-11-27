#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 11:11:03 2019

@author: li
"""
import numpy as np
from skimage import metrics
from sklearn.metrics import mean_absolute_error
import math

def MAE(img1, img2):
    mae_0=mean_absolute_error(img1[:,:,0], img2[:,:,0],
                              multioutput='uniform_average')
    mae_1=mean_absolute_error(img1[:,:,1], img2[:,:,1],
                              multioutput='uniform_average')
    mae_2=mean_absolute_error(img1[:,:,2], img2[:,:,2],
                              multioutput='uniform_average')
    return np.mean([mae_0,mae_1,mae_2])

def MSE_PSNR_SSIM(img1, img2):
    mse_ = np.mean( (img1 - img2) ** 2 )
    if mse_ == 0:
        return 100
    PIXEL_MAX = 1
    psnr = 10 * math.log10(PIXEL_MAX / mse_)
    return mse_, metrics.peak_signal_noise_ratio(img1, img2), metrics.structural_similarity(img1,
                                                    img2, data_range=PIXEL_MAX,
                                                    multichannel=True)
