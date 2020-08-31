import functools
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

import numpy as np

torch.set_num_threads(4)

from utils.utils import _get_bilinear_us_init_weights
from utils.utils import _get_xavier_init_weights

from Projection import RotateBox


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

class Decoder2DTower(nn.Module):
	'''
	Decoder Tower that takes a tensor and gives a 3D representation of it.
	'''
	def __init__(self, input_num_channels, num_output_channels=1):
		super(Decoder2DTower, self).__init__()

		self.us_1 = nn.ConvTranspose2d(input_num_channels, 512, 4)
		self.convt = nn.ConvTranspose2d(512, 256, 4, stride=2)
		self.convt0 = nn.ConvTranspose2d(256, 128, 5, stride=2)

		self.convt2 = nn.ConvTranspose2d(128, 64, 5, stride=2)
		self.convt3 = nn.ConvTranspose2d(64, num_output_channels, 6, stride=2, padding=1)

	def forward(self, x):
		x = F.relu(self.us_1(x))
		x = F.relu(self.convt(x))
		x = F.relu(self.convt0(x))
		x = F.relu(self.convt2(x))
		x = (self.convt3(x))
		
		return x

	def _initialize_weights(self):
		self.us_1._parameters['bias'].data.copy_(torch.zeros(self.us_1._parameters['bias'].size()))

		self.convt._parameters['bias'].data.copy_(torch.zeros(self.convt._parameters['bias'].size()))

		self.convt0._parameters['bias'].data.copy_(torch.zeros(self.convt0._parameters['bias'].size()))

		self.convt2._parameters['bias'].data.copy_(torch.zeros(self.convt2._parameters['bias'].size()))

		self.convt3._parameters['bias'].data.copy_(torch.zeros(self.convt3._parameters['bias'].size()))
		init.xavier_uniform(self.us_1._parameters['weight'], gain=init.calculate_gain('relu'))
		init.xavier_uniform(self.convt._parameters['weight'], gain=init.calculate_gain('relu'))
		init.xavier_uniform(self.convt0._parameters['weight'], gain=init.calculate_gain('relu'))
		init.xavier_uniform(self.convt2._parameters['weight'], gain=init.calculate_gain('relu'))
		init.kaiming_uniform(self.convt3._parameters['weight'])

class Decoder2DTowerRotateBox(nn.Module):
	def __init__(self, input_num_channels, num_output_channels=1, norm_type='batch'):
		super(Decoder2DTowerRotateBox, self).__init__()
		norm_layer = get_norm_layer(norm_type=norm_type)

		if type(norm_layer) == functools.partial:
			use_bias = norm_layer.func == nn.InstanceNorm2d or norm_type == 'none'
		else:
			use_bias = norm_layer == nn.InstanceNorm2d or norm_type == 'none'

		us_1 = nn.ConvTranspose3d(input_num_channels, 256, 4, stride=2, padding=1, bias=use_bias)
		convt = nn.ConvTranspose3d(256, 128, 4, stride=2, padding=1, bias=use_bias)
		convt0 = nn.ConvTranspose3d(128, 128, 4, stride=2, padding=1, bias=use_bias)

		convt2 = nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1, bias=use_bias)
		
		convt3 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=use_bias)
		
		convt4 = nn.ConvTranspose2d(64,64, 4, stride=2, padding=1, bias=use_bias)
		convt5 = nn.ConvTranspose2d(64,32, 4, stride=2, padding=1, bias=use_bias)
		convt6 = nn.ConvTranspose2d(32,num_output_channels, 4, stride=2, padding=1)

		self.rotatebox = RotateBox(8,8,8)

		self.us_1 = us_1
		self.norm1 = norm_layer(256)
		self.convt = convt
		self.norm2 = norm_layer(128)
		self.convt0 = convt0
		self.norm3 = norm_layer(128)
		self.convt2 = convt2
		self.norm4 = norm_layer(128)
		self.convt3 = convt3
		self.norm5 = norm_layer(64)
		self.convt4 = convt4
		self.norm6 = norm_layer(64)
		self.convt5 = convt5
		self.norm7 = norm_layer(32)
		self.convt6 = convt6

	def forward(self, x, theta):	
		x = x.unsqueeze(4)
		x = self.norm1(F.relu(self.us_1(x)))
		x = self.norm2(F.relu(self.convt(x)))
		x = self.norm3(F.relu(self.convt0(x)))
		x = self.rotatebox(x, theta)
		return x.m

class Decoder2DTowerLarge(nn.Module):
	'''
	Decoder Tower that takes a tensor and gives a 3D representation of it.
	'''
	def __init__(self, input_num_channels, num_output_channels=1, norm_type='batch', use_dropout=False):
		super(Decoder2DTowerLarge, self).__init__()
		norm_layer = get_norm_layer(norm_type=norm_type)

		if type(norm_layer) == functools.partial:
			use_bias = norm_layer.func == nn.InstanceNorm2d or norm_type == 'none'
		else:
			use_bias = norm_layer == nn.InstanceNorm2d or norm_type == 'none'


		us_1 = nn.ConvTranspose2d(input_num_channels, 256, 4, stride=2, padding=1, bias=use_bias)
		convt = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=use_bias)
		convt0 = nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1, bias=use_bias)

		convt2 = nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1, bias=use_bias)
		
		convt3 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=use_bias)
		
		convt4 = nn.ConvTranspose2d(64,64, 4, stride=2, padding=1, bias=use_bias)
		convt5 = nn.ConvTranspose2d(64,32, 4, stride=2, padding=1, bias=use_bias)
		convt6 = nn.ConvTranspose2d(32,num_output_channels, 4, stride=2, padding=1)

		if norm_type == 'batch' and use_dropout:
			sequence = [us_1, nn.LeakyReLU(0.2, True), norm_layer(256), nn.Dropout(0.5), \
						convt, nn.LeakyReLU(0.2, True), norm_layer(128), nn.Dropout(0.5), \
						convt0, nn.LeakyReLU(0.2, True), norm_layer(128), nn.Dropout(0.5), \
						convt2, nn.LeakyReLU(0.2, True), norm_layer(128), nn.Dropout(0.5), \
						convt3, nn.LeakyReLU(0.2, True), norm_layer(64), nn.Dropout(0.5), \
						convt4, nn.LeakyReLU(0.2, True), norm_layer(64), nn.Dropout(0.5), \
						convt5, nn.LeakyReLU(0.2, True), norm_layer(32), nn.Dropout(0.5), \
						convt6]
			self.model = nn.Sequential(*sequence)
		elif norm_type == 'batch':
			sequence = [us_1, nn.ReLU(True), norm_layer(256), \
						convt, nn.ReLU(True), norm_layer(128), \
						convt0, nn.ReLU(True), norm_layer(128), \
						convt2, nn.ReLU(True), norm_layer(128), \
						convt3, nn.ReLU(True), norm_layer(64), \
						convt4, nn.ReLU(True), norm_layer(64), \
						convt5, nn.ReLU(True), norm_layer(32), \
						convt6]
			self.model = nn.Sequential(*sequence)
		else:
			sequence = [us_1, nn.ReLU(True),  \
						convt, nn.ReLU(True), \
						convt0, nn.ReLU(True),  \
						convt2, nn.ReLU(True),  \
						convt3, nn.ReLU(True),  \
						convt4, nn.ReLU(True), \
						convt5, nn.ReLU(True),  \
						convt6]
			# self.us_1 = us_1
			# self.convt = convt
			# self.convt0 = convt0
			# self.convt2 = convt2
			# self.convt3 = convt3
			# self.convt4 = convt4
			# self.convt5 = convt5
			# self.convt6 = convt6
			self.model = nn.Sequential(*sequence)



	def forward(self, x):	
	# 	x = F.relu(self.us_1(x))
	# 	x = F.relu(self.convt(x))
	# 	x = F.relu(self.convt0(x))
	# 	x = F.relu(self.convt2(x))
	# 	x = F.relu(self.convt3(x))
	# 	x = F.relu(self.convt4(x))
	# 	x = F.relu(self.convt5(x))
	# 	return self.convt6(x)


		return self.model(x)

class Decoder2DTowerLargeMultiLevel(nn.Module):
	'''
	Decoder Tower that takes a tensor and gives a 3D representation of it.
	'''
	def __init__(self, input_num_channels, num_output_channels=1, norm_type='batch', use_dropout=False):
		super(Decoder2DTowerLargeMultiLevel, self).__init__()
		norm_layer = get_norm_layer(norm_type=norm_type)

		if type(norm_layer) == functools.partial:
			use_bias = norm_layer.func == nn.InstanceNorm2d or norm_type == 'none'
		else:
			use_bias = norm_layer == nn.InstanceNorm2d or norm_type == 'none'


		us_1 = nn.ConvTranspose2d(input_num_channels, 256, 4, stride=2, padding=1, bias=use_bias)
		convt = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=use_bias)
		convt0 = nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1, bias=use_bias)

		convt2 = nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1, bias=use_bias)
		
		convt3 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=use_bias)
		
		convt4 = nn.ConvTranspose2d(64,64, 4, stride=2, padding=1, bias=use_bias)
		convt5 = nn.ConvTranspose2d(64,32, 4, stride=2, padding=1, bias=use_bias)
		convt6 = nn.ConvTranspose2d(32,num_output_channels, 4, stride=2, padding=1)

		self.us_1 = us_1
		self.norm1 = norm_layer(256)
		self.convt = convt
		self.norm2 = norm_layer(128)
		self.convt0 = convt0
		self.norm3 = norm_layer(128)
		self.convt2 = convt2
		self.norm4 = norm_layer(128)
		self.convt3 = convt3
		self.norm5 = norm_layer(64)
		self.convt4 = convt4
		self.norm6 = norm_layer(64)
		self.convt5 = convt5
		self.norm7 = norm_layer(32)
		self.convt6 = convt6

		self.res1 = nn.Conv2d(32, 1, 3, stride=1, padding=1, bias=False)
		self.res2 = nn.Conv2d(64, 1, 3, stride=1, padding=1, bias=False)
		self.res3 = nn.Conv2d(64, 1, 3, stride=1, padding=1, bias=False)


	def forward(self, x):	
		x = self.norm1(F.relu(self.us_1(x)))
		x = self.norm2(F.relu(self.convt(x)))
		x = self.norm3(F.relu(self.convt0(x)))
		x = self.norm4(F.relu(self.convt2(x)))
		x2 = self.norm5(F.relu(self.convt3(x)))
		x1 = self.norm6(F.relu(self.convt4(x2)))
		x = self.norm7(F.relu(self.convt5(x1)))
		return self.convt6(x), self.res1(x), self.res2(x1), self.res3(x2)


class Decoder2DTowerResNet(nn.Module):
	'''
	Decoder Tower that takes a tensor and gives a 3D representation of it.
	'''
	def __init__(self, input_num_channels, num_output_channels=1):
		super(Decoder2DTowerResNet, self).__init__()
		self.us_1 = nn.ConvTranspose2d(input_num_channels, 512, 4)
		self.skip1 = nn.ConvTranspose2d(input_num_channels, 512, 4, bias=False)
		self.convta = nn.ConvTranspose2d(512, 256, 3, dilation=2)
		self.convtb = nn.ConvTranspose2d(256, 256, 3, dilation=2)

		self.bilinear2 = nn.UpsamplingBilinear2d((8,8))
		self.skip2 = nn.ConvTranspose2d(512, 256, 1)
		self.bilinear3 = nn.UpsamplingBilinear2d((19,19))
		self.skip3 = nn.ConvTranspose2d(256, 128, 1)
		self.bilinear4 = nn.UpsamplingBilinear2d((43,43))
		self.skip4 = nn.ConvTranspose2d(128, 64, 1)
		self.bilinear5 = nn.UpsamplingBilinear2d((89,89))
		self.skip5 = nn.ConvTranspose2d(64, 32, 1)
		self.bilinear6 = nn.UpsamplingBilinear2d((97,97))
		self.skip6 = nn.ConvTranspose2d(32, 32, 1)

		self.convt2a = nn.ConvTranspose2d(256, 128, 3, dilation=2, stride=2)
		self.convt2b = nn.ConvTranspose2d(128, 128, 3, dilation=2)
		self.convt3a = nn.ConvTranspose2d(128, 64, 3, dilation=2)
		self.convt3b = nn.ConvTranspose2d(64, 64, 3, dilation=2, stride=2)
		self.convt4a = nn.ConvTranspose2d(64, 32, 3, stride=2)
		self.convt4b = nn.ConvTranspose2d(32, 32, 3)
		self.convt5a = nn.ConvTranspose2d(32, 32, 3)
		self.convt5b = nn.ConvTranspose2d(32, 32, 3)
		self.convt6a = nn.ConvTranspose2d(32, 32, 3)
		self.convt6b = nn.ConvTranspose2d(32, 32, 3)
		self.convt7a = nn.ConvTranspose2d(32, num_output_channels, 4)

		self._initialize_weights()

	def forward(self, x):
		s1 = self.skip1(x)
		x = F.relu(self.us_1(x))
		x = x + s1
		s2 = self.skip2(self.bilinear2(x))

		x = F.relu(self.convtb(F.relu(self.convta(x))))
		
		x = x + s2
		s3 = self.skip3(self.bilinear3(x))
		x = F.relu(self.convt2b(F.relu(self.convt2a(x))))
		

		x = x + s3
		s4 = self.skip4(self.bilinear4(x))
		x = F.relu(self.convt3b(F.relu(self.convt3a(x))))
		
		x = x + s4
		s5 = self.skip5(self.bilinear5(x))
		
		x = self.convt4b(F.relu(self.convt4a(x)))
		
		x = x + s5
		s6 = self.skip6(self.bilinear6(x))
		x = self.convt5b(F.relu(self.convt5a(x)))
		x = self.convt6b(F.relu(self.convt6a(x)))
		x = x + s6
		x = self.convt7a(F.relu(x))
		return x

	def _initialize_weights(self):
		self.us_1._parameters['bias'].data.copy_(torch.zeros(self.us_1._parameters['bias'].size()))

		self.convta._parameters['bias'].data.copy_(torch.zeros(self.convta._parameters['bias'].size()))
		self.convtb._parameters['bias'].data.copy_(torch.zeros(self.convtb._parameters['bias'].size()))

		self.convt2a._parameters['bias'].data.copy_(torch.zeros(self.convt2a._parameters['bias'].size()))
		self.convt2b._parameters['bias'].data.copy_(torch.zeros(self.convt2b._parameters['bias'].size()))

		self.convt3a._parameters['bias'].data.copy_(torch.zeros(self.convt3a._parameters['bias'].size()))
		self.convt3b._parameters['bias'].data.copy_(torch.zeros(self.convt3b._parameters['bias'].size()))
		self.convt4a._parameters['bias'].data.copy_(torch.zeros(self.convt4a._parameters['bias'].size()))
		init.xavier_uniform(self.us_1._parameters['weight'], gain=init.calculate_gain('relu'))
		init.xavier_uniform(self.convta._parameters['weight'], gain=init.calculate_gain('relu'))
		init.xavier_uniform(self.convtb._parameters['weight'], gain=init.calculate_gain('relu'))
		init.xavier_uniform(self.convt2a._parameters['weight'], gain=init.calculate_gain('relu'))
		init.xavier_uniform(self.convt2b._parameters['weight'], gain=init.calculate_gain('relu'))
		init.xavier_uniform(self.convt3a._parameters['weight'], gain=init.calculate_gain('relu'))
		init.xavier_uniform(self.convt3b._parameters['weight'], gain=init.calculate_gain('relu'))



class Decoder3DTower(nn.Module):
	'''
	Decoder Tower that takes a tensor and gives a 3D representation of it.
	'''
	def __init__(self, input_num_channels):
		super(Decoder3DTower, self).__init__()


		self.convt1 = nn.ConvTranspose3d(input_num_channels, 256, 3, stride=2)
		self.conv1 = nn.Conv3d(256,256, 5)
		self.batchnorm1 = nn.BatchNorm3d(256)
		self.convt2 = nn.ConvTranspose3d(256, 128, 3, stride=2)
		self.conv2 = nn.Conv3d(128,128, 5)
		self.batchnorm2 = nn.BatchNorm3d(128)
		self.convt3 = nn.ConvTranspose3d(128, 64, 3, stride=2)
		self.conv3 = nn.Conv3d(64,64, 5)
		self.batchnorm3 = nn.BatchNorm3d(64)
		self.convt4 = nn.ConvTranspose3d(64, 1, 4, stride=2)

		self._initialize_weights()

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

		self.transformation = Transformation.PersTransformer()

		self._initialize_weights()

	def forward(self, x, res):
		x = F.relu(self.convt1(x))
		x = F.relu(self.convt2(x))
		x = F.relu(self.convt3(x))
		x = F.sigmoid(self.convt4(x))

		transformation = self.transformation(x, res)

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


