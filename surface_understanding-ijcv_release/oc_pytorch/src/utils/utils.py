import numpy as np
import torch
from torch.autograd import Variable

def get_bb(seg):
	# Get center of the bounding box of the segmentation
	bb_x = (seg.max(0)).nonzero()
	bb_x = (max(bb_x[0][0] - 1, 0), min(1 + bb_x[0][-1], seg.shape[0] - 1))
	bb_y = (seg.max(1)).nonzero()
	bb_y = (max(bb_y[0][0] - 1, 0), min(1 + bb_y[0][-1], seg.shape[0] - 1))
	return (bb_x, bb_y)
    
    
def get_xy_coordinates_np(w, batch_size):

	xy_all = np.stack((np.linspace(0,w-1,w).reshape((w,1)).repeat(w,1), 
					   np.linspace(0,w-1,w).reshape((w,1)).repeat(w,1).transpose()) 
                                , 2)

	return xy_all

def get_xy_coordinates(w, batch_size):

	xy_all = Variable(torch.Tensor(np.stack(
                                (np.linspace(0,w-1,w).reshape((w,1)).repeat(w,1), 
								 np.linspace(0,w-1,w).reshape((w,1)).repeat(w,1).transpose()) 
                                , 2))).cuda().unsqueeze(0).repeat(batch_size,1,1,1) / float(w) - 0.5

	return xy_all

def _get_bilinear_us_init_weights(weights):
	w = np.zeros(weights.size())
	k = weights.size(3)

	factor = np.floor((k + 1) / 2) 
	if np.remainder(k, 2) == 1:
		center = factor
	else:
		center = factor + 0.5

	C = range(1, k + 1)
	values = (np.ones((1,k)) - np.abs(C - center) / factor) 
	values = np.tile(np.dot(np.transpose(values), values), (3, 1, 1))
	values = np.transpose(values) * values

	for i in range(0, weights.size(1)):
		w[i,i,:,:,:] = values

	return torch.Tensor(w)

def _get_xavier_init_weights(weights):
	fan_out = weights.size(0)
	fan_in = np.prod(weights.size()[1:])
	sc = np.sqrt(2. / fan_in)

	w = np.random.randn(fan_out, fan_in) * sc

	return torch.Tensor(w)