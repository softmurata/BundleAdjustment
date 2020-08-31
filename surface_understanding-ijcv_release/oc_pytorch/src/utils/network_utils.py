# For operations on networks
import torch
from torch.autograd import Variable

def gradClamp(parameters, clip=5):
	max_value = torch.zeros(1,).cuda()
	min_value = torch.zeros(1,).cuda()
	for p in parameters:
		if not(p.grad is None):
			max_value = torch.Tensor([p.grad.data.max(max_value).max()]).cuda()
			min_value = torch.Tensor([p.grad.data.min(min_value).min()]).cuda()
			p.grad.data.clamp_(min=-clip, max=clip)

	return max_value.max(), min_value.min()