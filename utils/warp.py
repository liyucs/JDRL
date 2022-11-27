import os
import torch
from collections import OrderedDict
from abc import ABC, abstractmethod
# from . import networks
import math 
from utils.Flowfunction import flow_resample

backwarp_tenGrid = {}
backwarp_tenPartial = {}


def estimate(tenFirst, tenSecond, net):
	assert(tenFirst.shape[3] == tenSecond.shape[3])
	assert(tenFirst.shape[2] == tenSecond.shape[2])
	intWidth = tenFirst.shape[3]
	intHeight = tenFirst.shape[2]
	# tenPreprocessedFirst = tenFirst.view(1, 3, intHeight, intWidth)
	# tenPreprocessedSecond = tenSecond.view(1, 3, intHeight, intWidth)

	intPreprocessedWidth = int(math.floor(math.ceil(intWidth / 64.0) * 64.0))
	intPreprocessedHeight = int(math.floor(math.ceil(intHeight / 64.0) * 64.0))

	tenPreprocessedFirst = torch.nn.functional.interpolate(input=tenFirst, 
								size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)
	tenPreprocessedSecond = torch.nn.functional.interpolate(input=tenSecond, 
								size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)

	tenFlow = 20.0 * torch.nn.functional.interpolate(input=net(tenPreprocessedFirst, tenPreprocessedSecond), 
														size=(intHeight, intWidth), mode='bilinear', align_corners=False)

	tenFlow[:, 0, :, :] *= float(intWidth) / float(intPreprocessedWidth)
	tenFlow[:, 1, :, :] *= float(intHeight) / float(intPreprocessedHeight)

	return tenFlow[:, :, :, :]
	
def backwarp(tenInput, tenFlow):
	index = str(tenFlow.shape) + str(tenInput.device)
	if index not in backwarp_tenGrid:
		tenHor = torch.linspace(-1.0 + (1.0 / tenFlow.shape[3]), 1.0 - (1.0 / tenFlow.shape[3]), 
									tenFlow.shape[3]).view(1, 1, 1, -1).expand(-1, -1, tenFlow.shape[2], -1)
		tenVer = torch.linspace(-1.0 + (1.0 / tenFlow.shape[2]), 1.0 - (1.0 / tenFlow.shape[2]), 
									tenFlow.shape[2]).view(1, 1, -1, 1).expand(-1, -1, -1, tenFlow.shape[3])
		backwarp_tenGrid[index] = torch.cat([ tenHor, tenVer ], 1).to(tenInput.device)

	if index not in backwarp_tenPartial:
		backwarp_tenPartial[index] = tenFlow.new_ones([ tenFlow.shape[0], 
													1, tenFlow.shape[2], tenFlow.shape[3] ])

	tenFlow = torch.cat([ tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0), 
							tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0) ], 1)
	tenInput = torch.cat([ tenInput, backwarp_tenPartial[index] ], 1)

	tenOutput = torch.nn.functional.grid_sample(input=tenInput, 
					grid=(backwarp_tenGrid[index] + tenFlow).permute(0, 2, 3, 1), 
					mode='bilinear', padding_mode='zeros', align_corners=False)

	return tenOutput

# im_b/output -- ori-post

def get_flow(tenFirst, tenSecond, net, iftrain=False):
	if iftrain:
		flow = estimate(tenFirst, tenSecond, net) 
	else:
		with torch.no_grad():
			net.eval()
			flow = estimate(tenFirst, tenSecond, net) 
	return flow


def get_backwarp(im_b, im_gt, net, iftrain=False):

	flow = get_flow(im_b, im_gt, net, iftrain)
	wrap_gt = flow_resample(im_gt, flow)

	return wrap_gt, flow

