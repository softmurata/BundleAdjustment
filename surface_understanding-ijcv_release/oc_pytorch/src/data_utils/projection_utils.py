# projection utils
import torch
import torch.nn.functional as F
import numpy as np
import os
import torchvision
from torch.autograd import Variable
import math
import binvox_rw

import torchvision

SIZE=256
focal_length = math.sqrt(3.) * 2
#displacement = focal_length + math.sqrt(1)
# focal_length = math.sqrt(3.) * 2
# displacement = focal_length + math.sqrt(1) * 2

def unproject_shapenet(depthmap, theta, min_value=-9, size=128, phi=-30):
	points3dx = torch.linspace(5,-5,depthmap.size(3)).unsqueeze(0).repeat(size,1).unsqueeze(0).unsqueeze(0).cuda()
	points3dy = torch.linspace(5,-5,depthmap.size(3)).unsqueeze(1).repeat(1,size).unsqueeze(0).unsqueeze(0).cuda()

	points3d = torch.cat((depthmap+5, points3dx, points3dy, torch.ones(points3dy.size()).cuda()), 1).permute(0,2,3,1).contiguous().squeeze().view(-1,4).contiguous()
	points3d = points3d[points3d[:,0] > min_value+5, :]
	if points3d.size(0) == 0:
		points3d = torch.zeros((1,4)).cuda()

	c = - 10. / 20. + 9. / 4. 
	displacement =  focal_length + math.sqrt(1) * c 
	camera = get_camera_YXZ(theta, phi, focal_length, displacement).cuda()

	points3d = torch.mm(points3d, camera.transpose(1,0).contiguous())
	return torch.cat((points3d[:,1:2], points3d[:,2:3], points3d[:,0:1]), 1)

def unproject_sculptures(depthmap, theta, min_value=0.5, size=128):
	points3dx = torch.linspace(5,-5,depthmap.size(3)).unsqueeze(0).repeat(size,1).unsqueeze(0).unsqueeze(0).cuda()
	points3dy = torch.linspace(-5,5,depthmap.size(3)).unsqueeze(1).repeat(1,size).unsqueeze(0).unsqueeze(0).cuda()

	points3d = torch.cat((points3dx, depthmap, points3dy), 1).permute(0,2,3,1).contiguous().squeeze().view(-1,3).contiguous()
	points3d = points3d[points3d[:,1] > min_value, :]
	points3d[:,2] = points3d[:,2] - 3 # Subtract off centre
	rotation = torch.eye(3).cuda()
	rotation[0,0] = math.cos(-theta)
	rotation[1,1] = math.cos(-theta)
	rotation[0,1] = - math.sin(-theta)
	rotation[1,0] = math.sin(-theta)

	points3d = torch.mm(points3d, rotation.transpose(1,0).contiguous())
	return points3d.data


def generate_grid(focal_length):
	base_grid = torch.ones((SIZE,SIZE,SIZE,4))
	dmin = 1. / (focal_length + math.sqrt(3.)*4)
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

def project(grid, voxels, KRt, i=0):
	grid = grid.view(-1, 4).contiguous().cuda().permute(1,0).contiguous()
	grid3d = torch.matmul(KRt, grid)
	grid3d = grid3d.permute(1,0).contiguous().view(1,SIZE, SIZE,SIZE, 4)[:,:,:,:,0:3].contiguous()

	valid = F.grid_sample(voxels.unsqueeze(0).unsqueeze(0), grid3d) 
	valid = valid.view(1,1,SIZE,SIZE,SIZE)
	grid = grid.permute(1,0).contiguous().view(1,SIZE,SIZE,SIZE,4)[:,:,:,:,0:3]
	depths = grid[:,:,:,:,0].unsqueeze(0)
	depths[valid == 0] = np.inf

	depthx = depths.min(dim=2)[0]

	depthx[depthx > 1000] = 0
	# torchvision.utils.save_image(depthx / depthx.max(), '/scratch/local/ssd/ow/code/oc_pytorch/depth%d.png' % i)
	# torchvision.utils.save_image(voxels.squeeze().max(dim=0)[0], '/scratch/local/ssd/ow/code/oc_pytorch/depthx.png')
	# torchvision.utils.save_image(voxels.squeeze().max(dim=1)[0], '/scratch/local/ssd/ow/code/oc_pytorch/depthy.png')
	# torchvision.utils.save_image(voxels.squeeze().max(dim=2)[0], '/scratch/local/ssd/ow/code/oc_pytorch/depthz.png')
	return depthx.squeeze().cpu().data.numpy(), displacement

def get_camera_YXZ(phi, theta, focal_length, displacement):
	T = torch.zeros((4,4))
	K = torch.eye(4)
	E = torch.eye(4)

	sinphi = math.sin(phi * math.pi / 180.)
	cosphi = math.cos(phi * math.pi / 180.)
	sintheta = math.sin(theta * math.pi / 180.)
	costheta = math.cos(theta * math.pi / 180.)

	# Rotate about X
	Relet = torch.zeros((3,3))
	Relet[0,0] = costheta
	Relet[2,2] = costheta
	Relet[0,2] = -sintheta
	Relet[2,0] = sintheta
	Relet[1,1] = 1

	# Rotate about Y
	Razit = torch.zeros((3,3))
	Razit[0,0] = cosphi
	Razit[1,1] = cosphi
	Razit[0,1] = -sinphi
	Razit[1,0] = sinphi
	Razit[2,2] = 1

	Rcomb = torch.matmul(Razit, Relet)
	colR = torch.zeros((3,1))
	colR[0,0] = displacement
	colR = torch.matmul(Rcomb, colR)
	E[0:3,0:3] = Rcomb
	E[0:3,3:] = -colR

	K[2,2] = 1. / focal_length
	K[1,1] = 1. / focal_length
	return torch.matmul(E,K)

def get_camera(phi, theta, focal_length=math.sqrt(3.)/2.):
	T = torch.zeros((4,4))
	K = torch.eye(4)
	E = torch.eye(4)

	sinphi = math.sin(phi * math.pi / 180.)
	cosphi = math.cos(phi * math.pi / 180.)
	sintheta = math.sin(theta * math.pi / 180.)
	costheta = math.cos(theta * math.pi / 180.)

	# Rotate about X
	Relet = torch.zeros((3,3))
	Relet[1,1] = costheta
	Relet[2,2] = costheta
	Relet[1,2] = -sintheta
	Relet[2,1] = sintheta
	Relet[0,0] = 1

	# Rotate about Y
	Razit = torch.zeros((3,3))
	Razit[0,0] = cosphi
	Razit[1,1] = cosphi
	Razit[1,0] = -sinphi
	Razit[0,1] = sinphi
	Razit[2,2] = 1

	Rcomb = torch.matmul(Razit, Relet)
	colR = torch.zeros((3,1))
	colR[1,0] = focal_length + math.sqrt(1) / 2.
	colR = torch.matmul(Rcomb, colR)
	E[0:3,0:3] = Rcomb
	E[0:3,3:] = -colR

	K[0,0] = 1. / focal_length
	K[2,2] = 1. / focal_length
	return torch.matmul(E, K)

def normalize(voxels):
	minx = voxels.max(axis=2).max(axis=1)
	miny = voxels.max(axis=2).max(axis=0)
	minz = voxels.max(axis=0).max(axis=0)

	xs = np.where(minx > 0)[0]
	ys = np.where(miny > 0)[0]
	zs = np.where(minz > 0)[0]
	(minx, maxx) = (xs.min(), xs.max())
	(miny, maxy) = (ys.min(), ys.max())
	(minz, maxz) = (zs.min(), zs.max())

	maxd = max(max((maxx - minx), (maxy - miny)), (maxz - minz))
	voxels = voxels[minx:maxx+1,miny:maxy+1,minz:maxz+1]
	padxb = int((maxd - (maxx-minx)) / 2)
	padxa = int((maxd - (maxx-minx)+1) / 2)
	padyb = int((maxd - (maxy-miny)) / 2)
	padya = int((maxd - (maxy-miny)+1) / 2)
	padzb = int((maxd - (maxz-minz)) / 2)
	padza = int((maxd - (maxz-minz)+1) / 2)

	# print(((padxb, padxa), (padyb, padya), (padzb, padza)))
	maxpad = max(max(padxb, padyb), padzb)
	maxpad=10
	print(maxpad)
	c = - maxpad / 20. + 9. / 4. 
	print(c)
	displacement =  focal_length + math.sqrt(1) * c
	voxels = np.pad(voxels, ((padxb, padxa), (padyb, padya), (padzb, padza)), 'constant')
	# voxels = np.pad(voxels, ((5, 5), (0, 0), (5, 5)), 'constant')
	return voxels, displacement

if __name__ == '__main__':
	grid = generate_grid(focal_length)
	i = 0
	for file in os.listdir('/scratch/shared/slow/ow/shapenet/ShapeNetCore.v2/03001627/'):
		if os.path.exists('/scratch/shared/slow/ow/shapenet/ShapeNetCore.v2/03001627/' + file + '/models/model_normalized.surface.binvox'):
			file_path = '/scratch/shared/slow/ow/shapenet/ShapeNetCore.v2/03001627/' + file + '/models/model_normalized.surface.binvox'
			voxels = binvox_rw.read_as_3d_array(open(file_path, 'rb')).data.astype(np.float32)
			voxels, displacement = normalize(voxels)

			#Gives YZX coordinate frame
			voxels = voxels.transpose(1,2,0)
			voxels = np.flip(voxels, 0).copy()
			voxels = np.flip(voxels, 1).copy()
			voxels = torch.Tensor(voxels).cuda()
			voxels = Variable(voxels)

			results = np.zeros((24,SIZE,SIZE))

			for camera_id in range(1,25):
				camera = get_camera_YXZ(270-camera_id*15, 30, focal_length, displacement)
				camera = Variable(torch.Tensor(camera)).cuda()

				depth = project(grid, voxels, camera, camera_id)
				results[camera_id-1,:,:] = depth[0]

			print(file_path)
			if not os.path.exists('/scratch/local/ssd/ow/shapenet_nips/depth/03001627/' + file):
				os.makedirs('/scratch/local/ssd/ow/shapenet_nips/depth/03001627/' + file)

			np.savez_compressed('/scratch/local/ssd/ow/shapenet_nips/depth/03001627/' + file + '/depth_maps', displacement=np.array([displacement]), 
				depthmaps=results)
		else:
			print("WARNING: " + file + " does not exist.")