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
        init.xavier_normal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.xavier_normal(m.weight.data, gain=1)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)

class Decoder2DTower(nn.Module):
	'''
	Decoder Tower that takes a tensor and gives a 3D representation of it.
	'''
	def __init__(self, input_num_channels):
		super(Decoder2DTower, self).__init__()
		self.us_1 = nn.ConvTranspose2d(input_num_channels, 256, 4, stride=1)
		self.convt = nn.ConvTranspose2d(256, 128, 5, stride=2, padding=2)
		self.convt0 = nn.ConvTranspose2d(128, 128, 5, stride=2, padding=1)

		self.convt2 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1)
		self.conv2 = nn.Conv2d(64,64,5,padding=2)
		self.convt3 = nn.ConvTranspose2d(64, 32, 3, stride=2,padding=1)
		self.conv3 = nn.Conv2d(32,1,5,padding=2)

		init_weights(self)

	def forward(self, x):
		x = F.relu(self.us_1(x))
		x = F.relu(self.convt(x))
		x = F.relu(self.convt0(x))
		x = F.relu(self.conv2(F.relu(self.convt2(x))))
		x = self.conv3(F.relu(self.convt3(x)))
		#x = F.sigmoid(x)
		
		return x

	def _initialize_weights(self):
		self.us_1._parameters['weight'].data.copy_(_get_xavier_init_weights(self.us_1._parameters['weight']))
		self.us_1._parameters['bias'].data.copy_(torch.zeros(self.us_1._parameters['bias'].size()))

		self.convt._parameters['weight'].data.copy_(_get_xavier_init_weights(self.convt._parameters['weight']))
		self.convt._parameters['bias'].data.copy_(torch.zeros(self.convt._parameters['bias'].size()))

		self.convt0._parameters['weight'].data.copy_(_get_xavier_init_weights(self.convt0._parameters['weight']))
		self.convt0._parameters['bias'].data.copy_(torch.zeros(self.convt0._parameters['bias'].size()))

		self.convt2._parameters['weight'].data.copy_(_get_xavier_init_weights(self.convt2._parameters['weight']))
		self.convt2._parameters['bias'].data.copy_(torch.zeros(self.convt2._parameters['bias'].size()))

		self.convt3._parameters['weight'].data.copy_(_get_xavier_init_weights(self.convt3._parameters['weight']))
		self.convt3._parameters['bias'].data.copy_(torch.zeros(self.convt3._parameters['bias'].size()))

		self.conv2._parameters['weight'].data.copy_(_get_xavier_init_weights(self.conv2._parameters['weight']))
		self.conv2._parameters['bias'].data.copy_(torch.zeros(self.conv2._parameters['bias'].size()))

		self.conv3._parameters['weight'].data.copy_(_get_xavier_init_weights(self.conv3._parameters['weight']))
		self.conv3._parameters['bias'].data.copy_(torch.zeros(self.conv3._parameters['bias'].size()))

class Decoder2DTowerLarge(nn.Module):
	'''
	Decoder Tower that takes a tensor and gives a 3D representation of it.
	'''
	def __init__(self, input_num_channels):
		super(Decoder2DTower, self).__init__()
		self.us_1 = nn.ConvTranspose2d(input_num_channels, 256, 4, stride=1)
		self.convt = nn.ConvTranspose2d(256, 128, 5, stride=2, padding=2)
		self.convt0 = nn.ConvTranspose2d(128, 128, 5, stride=2, padding=1)

		self.convt2 = nn.ConvTranspose2d(128, 128, 5, stride=2, padding=1)
		
		self.convt3 = nn.ConvTranspose2d(128, 64, 5, stride=2,padding=1)
		
		self.convt4 = nn.ConvTranspose2d(64,32,5,stride=2,padding=1)
		self.convt5 = nn.ConvTranspose2d(32,1,5,stride=2,padding=1)

		self._initialize_weights()

	def forward(self, x):
		x = F.relu(self.us_1(x))
		x = F.relu(self.convt(x))
		x = F.relu(self.convt0(x))
		x = F.relu(self.convt2(x))
		x = F.relu(self.convt3(x))
		x = F.relu(self.convt4(x))
		x = F.relu(self.convt5(x))
		x = F.tanh(x) * 10
		
		return x

	def _initialize_weights(self):
		self.us_1._parameters['weight'].data.copy_(_get_xavier_init_weights(self.us_1._parameters['weight']))
		self.us_1._parameters['bias'].data.copy_(torch.zeros(self.us_1._parameters['bias'].size()))

		self.convt._parameters['weight'].data.copy_(_get_xavier_init_weights(self.convt._parameters['weight']))
		self.convt._parameters['bias'].data.copy_(torch.zeros(self.convt._parameters['bias'].size()))

		self.convt0._parameters['weight'].data.copy_(_get_xavier_init_weights(self.convt0._parameters['weight']))
		self.convt0._parameters['bias'].data.copy_(torch.zeros(self.convt0._parameters['bias'].size()))

		self.convt2._parameters['weight'].data.copy_(_get_xavier_init_weights(self.convt2._parameters['weight']))
		self.convt2._parameters['bias'].data.copy_(torch.zeros(self.convt2._parameters['bias'].size()))

		self.convt3._parameters['weight'].data.copy_(_get_xavier_init_weights(self.convt3._parameters['weight']))
		self.convt3._parameters['bias'].data.copy_(torch.zeros(self.convt3._parameters['bias'].size()))

		self.convt4._parameters['weight'].data.copy_(_get_xavier_init_weights(self.convt4._parameters['weight']))
		self.convt4._parameters['bias'].data.copy_(torch.zeros(self.convt4._parameters['bias'].size()))

		self.convt5._parameters['weight'].data.copy_(_get_xavier_init_weights(self.convt5._parameters['weight']))
		self.convt5._parameters['bias'].data.copy_(torch.zeros(self.convt5._parameters['bias'].size()))

