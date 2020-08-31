import sys
import os
sys.path.append('./oc_pytorch/src/')
sys.path.append('./oc_pytorch/src/projection')

import numpy as np
import cv2 
np.random.seed(0)
EPS = 1e-5
DEBUG=False
import torch.nn as nn
import torch
from torch.autograd import Variable, Function
import torch.nn.functional as F
from utils.statistics import _get_iou_mask_accuracy_for_tensor
from scipy.ndimage.morphology import distance_transform_edt
from Decoders import Decoder2DTowerLarge
from Projection import Projection, RotateBox, PTN
from Decoder2DTower import Decoder2DTower
from Decoder3D import Decoder3DTower
torch.manual_seed(0)
torch.cuda.manual_seed(0)


torch.set_num_threads(1)


from utils.Globals import *
class AngleEncoder(nn.Module):
	def __init__(self):
		super(AngleEncoder, self).__init__()
		self.compute_stats = False

		# Pass through some non-linearities
		self.fv_1 = nn.Conv2d(2, 32, 1)  # cos(theta), sin(theta)
		self.fv_2 = nn.Conv2d(32, 1024, 1)

	def forward(self, theta):
		return F.relu(self.fv_2(F.relu(self.fv_1(theta)))).view(theta.size(0), 512,2,1).contiguous()



class SilhouettePrediction(nn.Module):
	def __init__(self, num_input_channels=512, num_output_channels=1, additional_param=3):
		super(SilhouettePrediction, self).__init__()
		self.compute_stats = False
  
        # upper string exists in utils.Globals.py
        # In this case, USE_SHAPENET and SMALL_DECODER_3D case
        
        
        # Input channels => 512
		if USE_SHAPENET and SMALL_DECODER_3D:
			self.extra_non_lin = nn.Conv2d(num_input_channels, num_input_channels, 1, bias=True)
			self.PTN = PTN(57,57,57)
			self.decoder = Decoder3DTower(num_input_channels)
		elif SMALL_DECODER_3D:
			self.extra_non_lin = nn.Conv2d(num_input_channels, num_input_channels, 1, bias=True)
			self.rotate_box = RotateBox(57,57,57)
			self.decoder = Decoder3DTower(num_input_channels)
			print("SMALL_DECODER_3D")
		elif SMALL_DECODER:
			self.decoder = Decoder2DTower(num_input_channels+32)
			print("SMALL_DECODER")
	
		else:
			self.decoder = Decoder2DTowerLarge(num_input_channels+32, num_output_channels=num_output_channels, norm_type='batch') 
			print("NORMAL_DECODER")

		# Pass through some non-linearities
			self.fv_1 = nn.Conv2d(2, 32, 1)
			self.fv_2 = nn.Conv2d(32, 32, 1)

	def forward(self, theta, fv_height, ref_seg):
		null_vector = F.relu(self.fv_2(F.relu(self.fv_1(theta))))

		if USE_SHAPENET and SMALL_DECODER_3D:
			fv_height = F.relu(self.extra_non_lin(fv_height))
			xc = self.decoder(fv_height.unsqueeze(4))
			theta = torch.atan2(theta[:,1,:,:], theta[:,0,:,:])
			xc = self.PTN(xc, theta)
			xc = xc.max(dim=2)[0]

		elif SMALL_DECODER_3D:
			fv_height = F.relu(self.extra_non_lin(fv_height))
			xc = self.decoder(fv_height.unsqueeze(4))
			theta = torch.atan2(theta[:,1,:,:], theta[:,0,:,:])
			xc = Projection()(xc, theta)
			xc = xc.min(3)[0].squeeze(3).transpose(3,2)
			ref_seg = 1 - ref_seg
		
		else:
			xc = self.decoder(torch.cat((fv_height, null_vector), 1))


		# Disregard those without silhouettes
		if (WEIGHT_MINMAX) and not(self.compute_stats):
			weights = np.zeros(ref_seg.data.cpu().numpy().shape)
			for i in range(0, ref_seg.size(0)):
				weightsinside = distance_transform_edt(ref_seg.data[i,0,:,:].cpu().numpy())
				weightsoutside = distance_transform_edt((1 - ref_seg.data[i,0,:,:]).cpu().numpy())
				weights[i,:,:,:] = weightsinside + weightsoutside

			if WEIGHT_MINMAX:
				weights[weights > 20] = 5
			
			weights = torch.Tensor(weights).clamp(min=0, max=20)


		if self.compute_stats:
			accuracy, total_num, acc_by_id = _get_iou_mask_accuracy_for_tensor(F.sigmoid(xc), ref_seg)
			
			return nn.BCEWithLogitsLoss().cuda()(xc, Variable(xc.data).cuda()), accuracy / total_num, Variable(F.sigmoid(xc).data), acc_by_id
		else:
			return nn.BCEWithLogitsLoss().cuda()(xc, ref_seg), Variable(F.sigmoid(xc).data)

