import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

import numpy as np

from utils_weights import _get_bilinear_us_init_weights
from utils_weights import _get_xavier_init_weights

def init_weights(net):
    net.apply(weights_init_xavier)

def weights_init_xavier(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.normal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.normal(m.weight.data, gain=1)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)

class Decoder3DTower(nn.Module):
	'''
	Decoder Tower that takes a tensor and gives a 3D representation of it.
	'''
	def __init__(self, input_num_channels):
		super(Decoder3DTower, self).__init__()


		self.convt1 = nn.ConvTranspose3d(input_num_channels, 256, 5)
		self.conv1 = nn.Conv3d(256,256, 5)
		self.batchnorm1 = nn.BatchNorm3d(256)
		self.convt2 = nn.ConvTranspose3d(256, 128, 5, stride=2)
		self.conv2 = nn.Conv3d(128,128, 5)
		self.batchnorm2 = nn.BatchNorm3d(128)
		self.convt3 = nn.ConvTranspose3d(128, 64, 5, stride=2, padding=1)
		self.conv3 = nn.Conv3d(64,64, 5)
		self.batchnorm3 = nn.BatchNorm3d(64)
		self.convt4 = nn.ConvTranspose3d(64, 1, 5, stride=2)


	def forward(self, x):
		x = F.relu(self.convt1(x))
		x = F.relu(self.convt2(x))
		x = F.relu(self.convt3(x))
		x = self.convt4(x)
		return x

	def _initialize_weights(self):
		self.convt1._parameters['weight'].data.copy_(_get_xavier_init_weights(self.convt1._parameters['weight']))
		self.convt1._parameters['bias'].data.copy_(torch.zeros(self.convt1._parameters['bias'].size()))

		self.convt2._parameters['weight'].data.copy_(_get_xavier_init_weights(self.convt2._parameters['weight']))
		self.convt2._parameters['bias'].data.copy_(torch.zeros(self.convt2._parameters['bias'].size()))

		self.convt3._parameters['weight'].data.copy_(_get_xavier_init_weights(self.convt3._parameters['weight']))
		self.convt3._parameters['bias'].data.copy_(torch.zeros(self.convt3._parameters['bias'].size()))

		self.convt4._parameters['weight'].data.copy_(_get_xavier_init_weights(self.convt4._parameters['weight']))
		self.convt4._parameters['bias'].data.copy_(torch.zeros(self.convt4._parameters['bias'].size()))

		self.conv1._parameters['weight'].data.copy_(_get_xavier_init_weights(self.conv1._parameters['weight']))
		self.conv1._parameters['bias'].data.copy_(torch.zeros(self.conv1._parameters['bias'].size()))

		self.conv2._parameters['weight'].data.copy_(_get_xavier_init_weights(self.conv2._parameters['weight']))
		self.conv2._parameters['bias'].data.copy_(torch.zeros(self.conv2._parameters['bias'].size()))

		self.conv3._parameters['weight'].data.copy_(_get_xavier_init_weights(self.conv3._parameters['weight']))
		self.conv3._parameters['bias'].data.copy_(torch.zeros(self.conv3._parameters['bias'].size()))


class Decoder3DTowerSmall(nn.Module):
	'''
	Decoder Tower that takes a tensor and gives a 3D representation of it.
	'''
	def __init__(self, input_num_channels):
		super(Decoder3DTowerSmall, self).__init__()

		self.convt1 = nn.ConvTranspose3d(input_num_channels, 256, 3)
		self.batchnorm1 = nn.BatchNorm3d(256)
		self.convt2 = nn.ConvTranspose3d(256, 128, 4)
		self.batchnorm2 = nn.BatchNorm3d(128)
		self.convt3 = nn.ConvTranspose3d(128, 64, 5, stride=2)
		self.batchnorm3 = nn.BatchNorm3d(64)
		self.convt4 = nn.ConvTranspose3d(64, 1, 6, stride=2, padding=1)

		self._initialize_weights()

	def forward(self, x):
		x = self.batchnorm1(F.relu(self.convt1(x)))
		x = self.batchnorm2(F.relu(self.convt2(x)))
		x = self.batchnorm3(F.relu(self.convt3(x)))
		x = self.convt4(x)
		return x

	def _initialize_weights(self):
		self.convt1._parameters['weight'].data.copy_(_get_xavier_init_weights(self.convt1._parameters['weight']))
		self.convt1._parameters['bias'].data.copy_(torch.zeros(self.convt1._parameters['bias'].size()))

		self.convt2._parameters['weight'].data.copy_(_get_xavier_init_weights(self.convt2._parameters['weight']))
		self.convt2._parameters['bias'].data.copy_(torch.zeros(self.convt2._parameters['bias'].size()))

		self.convt3._parameters['weight'].data.copy_(_get_xavier_init_weights(self.convt3._parameters['weight']))
		self.convt3._parameters['bias'].data.copy_(torch.zeros(self.convt3._parameters['bias'].size()))

		self.convt4._parameters['weight'].data.copy_(_get_xavier_init_weights(self.convt4._parameters['weight']))
		self.convt4._parameters['bias'].data.copy_(torch.zeros(self.convt4._parameters['bias'].size()))


