import numpy as np
import torch

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