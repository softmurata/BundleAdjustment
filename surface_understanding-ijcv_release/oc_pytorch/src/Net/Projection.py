import os

import torch.nn as nn
import torch.nn.functional as F
import torch

import math
import numpy as np
from torch.autograd import Variable

from torch.autograd import Function


from projection.modules.stn import VTN
from projection.modules.gridgen import PersGridGen




def _rotate_fast_bilinear_onetheta(x1, dTheta):
	# Rotate according to a number of thetas (one for each batch)
	assert(x1.shape[0] % 2 == 1)
	numpy_input = x1
	
	width = numpy_input.shape[1]-1
	height = numpy_input.shape[2]-1
	
	center = [np.floor((width+1) / 2.), np.floor((height+1) / 2.)]
	
	R = np.array([[np.cos(dTheta), -np.sin(dTheta)], [np.sin(dTheta), np.cos(dTheta)]])
	i = np.ravel(np.tile(np.linspace(0, width, width+1), (height+1, 1)))
	j = np.ravel(np.transpose(np.tile(np.linspace(0, height, height+1), (width+1, 1))))
	
	i_c = i - center[0] 
	j_c = j - center[1]
	i_prime = np.multiply(R[0,0], i_c) + np.multiply(R[0,1], j_c)  + center[0] 
	j_prime = np.multiply(R[1,0], i_c) + np.multiply(R[1,1], j_c) + center[1] 



	# Could go off the grid
	i_prime = np.minimum(i_prime, width)
	j_prime = np.minimum(j_prime, height)
	i_prime = np.maximum(i_prime, 0)
	j_prime = np.maximum(j_prime, 0)

	i_prime_0 = 1 - (i_prime - np.floor(i_prime))
	j_prime_0 = 1 - (j_prime - np.floor(j_prime))

	if len(x1.shape) == 4:
		i_prime_0 = i_prime_0[:,newaxis,newaxis]
		j_prime_0 = j_prime_0[:,newaxis, newaxis]
	else:
		i_prime_0 = i_prime_0[:,newaxis]
		j_prime_0 = j_prime_0[:,newaxis]


	i_prime = np.floor(i_prime).astype('int')
	j_prime = np.floor(j_prime).astype('int')
	i_primep = np.minimum(i_prime+1, width)
	j_primep = np.minimum(j_prime+1, height)

	output = numpy_input[:,i_prime, j_prime,:] * i_prime_0 * j_prime_0 + \
				numpy_input[:,i_prime, j_primep,:] * i_prime_0 * (1 - j_prime_0) + \
					numpy_input[:,i_primep, j_prime,:] * (1 - i_prime_0) * j_prime_0 + \
					numpy_input[:,i_primep, j_primep,:] * (1 - i_prime_0) * (1 - j_prime_0)

	output = output.reshape(x1.shape).transpose(0,2,1,3)
	return output

def _rotate_fast_bilinear(x1, dThetas):
	output = torch.Tensor(numpy_input.size())

	for batch in range(0, x1.size(0)):
		output[batch,:,:,:,:] = torch.Tensor(_rotate_fast_bilinear_onetheta(x1[batch,:,:,:].cpu().data.numpy(), dThetas[batch][0].cpu().data.numpy()))

	if int(os.environ['CUDA']) == 1:
		return output.cuda()
	else:
		return output

def _rotate_fast(x1, dThetas):
	# Rotate according to a number of thetas (one for each batch)
	assert(x1.size(0) == dThetas.size(0))
	numpy_input = x1
	output = torch.Tensor(numpy_input.size())
	
	channels = numpy_input.size(1)
	width = numpy_input.size(2)-1
	height = numpy_input.size(3)-1
	depth = numpy_input.size(4)-1
	
	center = [np.floor((width+1) / 2.), np.floor((height+1) / 2.)]
	for batch in range(0, x1.size(0)):
		dTheta = dThetas[batch][0]
		R = np.array([[np.cos(dTheta), -np.sin(dTheta)], [np.sin(dTheta), np.cos(dTheta)]])
		i = np.ravel(np.tile(np.linspace(0, width, width+1), (height+1, 1)))
		j = np.ravel(np.transpose(np.tile(np.linspace(0, height, height+1), (width+1, 1))))
		
		i_c = i - center[0] 
		j_c = j - center[1]
		i_prime = np.floor(np.multiply(R[0,0], i_c) + np.multiply(R[0,1], j_c)  + center[0] + 0.5)
		j_prime = np.floor(np.multiply(R[1,0], i_c) + np.multiply(R[1,1], j_c) + center[1] + 0.5)


		# Could go off the grid
		i_prime = np.minimum(i_prime, width)
		j_prime = np.minimum(j_prime, height)
		i_prime = np.maximum(i_prime, 0)
		j_prime = np.maximum(j_prime, 0)
		if int(os.environ['CUDA']) == 1:
			indexes = torch.Tensor(i_prime+j_prime*(width+1)).long().cuda()
		else:
			indexes = torch.Tensor(i_prime+j_prime*(width+1)).long()
		
		output[batch,:,:,:,:] = numpy_input[batch,:,:,:,:].view(channels, (width+1)*(height+1), depth+1).index_select(1, indexes).view(channels, width+1, height+1, depth+1)

	if int(os.environ['CUDA']) == 1:
		return output.cuda()
	else:
		return output

def generate_grid(focal_length, SIZE):
	base_grid = torch.ones((SIZE,SIZE,SIZE,4))
	dmin = 1. / (focal_length*3)
	dmax = 1. / focal_length

	print("GENERATING GRID....")

	for k in range(0, (SIZE)):
		for i in range(0, (SIZE)):
			for j in range(0, (SIZE)):
				disf = dmin + float(k) / (SIZE-1.) * (dmax - dmin)
				base_grid[k, i, j, 0] = 1. / disf
				base_grid[k, i, j, 2] = (-1. + float(i) / ((SIZE-1.)) * 2.) / disf
				base_grid[k, i, j, 1] = (-1. + float(j) / ((SIZE-1.)) * 2.) / disf

	return base_grid

class PTN(nn.Module):
	def __init__(self, height, width, depth, focal_length=math.sqrt(3.) * 2., SIZE=57):
		super(PTN, self).__init__()
		self.base_grid = generate_grid(focal_length, SIZE).cuda()
		self.SIZE = SIZE


	def project(self, voxels, KRt):
        # KRt => K.dot([R|t])
        b = voxels.size(0) # batch size 
		grid = self.base_grid.view(-1, 4).contiguous().cuda().permute(1,0).contiguous().unsqueeze(0).repeat(b, 1, 1)
		grid3d = torch.bmm(KRt, grid)
		grid3d = grid3d.permute(0,2,1).contiguous().view(b,self.SIZE, self.SIZE,self.SIZE, 4)[:,:,:,:,0:3].contiguous()
		projection = F.grid_sample(voxels, grid3d.detach(), padding_mode='border') 
		projection = projection.view(b,1,self.SIZE,self.SIZE,self.SIZE)  #(batch size, 1, height, width, depth)

		return projection.contiguous()

	def get_camera_YXZ(self, phi, theta=30, focal_length=math.sqrt(3.)*2, displacement=math.sqrt(3.)*2-0.5+9./4.):
		b = phi.size(0)

		T = torch.zeros((b, 4,4))
		K = torch.eye(4).unsqueeze(0).repeat(b, 1, 1)
		E = torch.eye(4).unsqueeze(0).repeat(b, 1, 1)

		sinphi = phi.sin()
		cosphi = phi.cos()
		sintheta = np.sin(theta * math.pi / 180.)
		costheta = np.cos(theta * math.pi / 180.)

		# Rotate about X
		Relet = torch.zeros((b,3,3))
		Relet[:,0,0] = costheta
		Relet[:,2,2] = costheta
		Relet[:,0,2] = -sintheta
		Relet[:,2,0] = sintheta
		Relet[:,1,1] = 1

		# Rotate about Y
		Razit = torch.zeros((b,3,3))
		Razit[:,0,0] = cosphi.squeeze()
		Razit[:,1,1] = cosphi.squeeze()
		Razit[:,0,1] = -sinphi.squeeze()
		Razit[:,1,0] = sinphi.squeeze()
		Razit[:,2,2] = 1

		Rcomb = torch.bmm(Razit, Relet)
		colR = torch.zeros((b,3,1))
		colR[:,0,0] = displacement
		colR = torch.bmm(Rcomb, colR)
		E[:,0:3,0:3] = Rcomb
		E[:,0:3,3:] = -colR

		K[:,2,2] = 1. / focal_length
		K[:,1,1] = 1. / focal_length
		return torch.bmm(E,K).cuda()

	def forward(self, voxels, theta):
		krt = self.get_camera_YXZ(theta)
		return self.project(voxels, krt)


class RotateBox(nn.Module):
	def __init__(self, height, width, depth):
		super(RotateBox, self).__init__()

		self.affinegrid = PersGridGen(height=height, width=width, depth=depth)


		self.vtn = VTN()

	def forward(self, input1, theta1):
		self.affinerotation = Variable(torch.Tensor(np.eye(3,4))).cuda()
		self.affinerotation = self.affinerotation.unsqueeze(0)
		self.affinerotation = self.affinerotation.repeat(input1.size(0), 1, 1)
		self.affinerotation[:,0,0] = theta1[:,1,:]
		self.affinerotation[:,0,2] = -theta1[:,0,:]
		self.affinerotation[:,2,0] = theta1[:,0,:]
		self.affinerotation[:,2,2] = theta1[:,1,:]
		
		affinegrid = self.affinegrid(self.affinerotation)
		#print(input1.min().cpu().data[0], input1.max().cpu().data[0])
		#print(input1.size())
		res = self.vtn(input1.permute(0,2,3,4,1), affinegrid)
		# Take mean over y axis
		#print(res.size())
		res = res.max(dim=3)[0]
		print('max_3, permute_3,2,1, backward')

		# should be either max_1, permute 3x1x2 or max_3 permute 3x2x1
		
		return res.permute(0,3,2,1)


class Projection(Function):
    def forward(self, x1, dThetas):
        x1 = _rotate_fast(x1, dThetas)
        self.thetas = dThetas
        return x1

    def backward(self, grad_output, retain_variables=True):
        return _rotate_fast(grad_output, -self.thetas), 0 * self.thetas
