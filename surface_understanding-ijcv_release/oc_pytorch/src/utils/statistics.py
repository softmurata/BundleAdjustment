import torch
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
from scipy import ndimage
from PIL import Image
from  Globals import *
import os
EPSILON = 0.001

torch.set_num_threads(1)


def _get_iou_mask_accuracy_for_tensor(tensor, masks):
	if not(USE_SHAPENET):
		tensor = nn.Upsample(size=(256, 256), mode='bilinear').cuda()(tensor)
	else:
		tensor = nn.Upsample(size=(32, 32), mode='bilinear').cuda()(tensor)

	if masks[0,0,0,0].cpu().data.item() == 1:
		intersection = ((tensor < 0.5).float() * (1 - masks)).float()
		union = (((tensor < 0.5).float() + (1 - masks)) > 0.9).float()

	else:
		intersection = ((tensor > 0.5).float() * masks).float()
		union = (((tensor > 0.5).float() + masks) > 0.9).float()
	


	total_iou = 0
	total_num = 0
	ious = torch.zeros((tensor.size(0), 1)).cuda()
	for b in range(0, tensor.size(0)):
		if (union[b,:,:,:].sum().cpu().data.numpy().item() > EPSILON):
			total_iou = total_iou + (intersection[b,:,:,:].sum() / union[b,:,:,:].sum())
			ious[b,:] = (intersection[b,:,:,:].sum() / union[b,:,:,:].sum())
			if np.isnan(total_iou.cpu().data.numpy().item()):
				print(1+'1')
			total_num = total_num + 1


	if 	total_num == 0:
		return 1, 1
	return total_iou.cpu().data.numpy().item(), total_num, ious

