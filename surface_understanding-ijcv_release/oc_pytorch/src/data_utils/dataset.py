import scipy.io
import numpy as np
import torch
from torchvision.transforms import Compose, ToTensor, Resize, CenterCrop
import os

from load_functions import load_any_img, load_img, get_sin_cos
from binvox_rw import read_as_3d_array
import cv2

from PIL import Image

import torch
from torch.autograd import Variable
import torch.utils.data as data

torch.set_num_threads(4)
BASE_FILE_TO_DIR = ''
NIPS_LOCATION='/scratch/shared/slow/ow/shapenet/nips/nips16_PTN/data/'

def get_avg_images(file_paths, base_directory='', suffix='', max_num=1):
	mean_r = 0
	mean_g = 0
	mean_b = 0

	indices = np.random.choice(range(0, len(file_paths)), min(max_num, len(file_paths)), replace=False)
	num_indices = 0
	print("Number of images used to compute mean: %d" % len(indices))
	for i in range(0, len(indices)):
		if len(file_paths[indices[i]]) == 1:
			file_path = base_directory + str(file_paths[indices[i]][0]) + suffix
		else:
			file_path = base_directory + str(file_paths[indices[i]]) + suffix

		if not os.path.exists(file_path):
			continue

		num_indices = num_indices + 1
		if i % 100 == 0:
			print((i, len(file_paths)))
		r,g,b = Image.open(file_path).convert('RGB').split()
		mean_b = mean_b + np.array(b.getdata()).mean()
		mean_g = mean_g + np.array(g.getdata()).mean()
		mean_r = mean_r + np.array(r.getdata()).mean()

	mean = mean_r / num_indices, mean_b / num_indices, mean_g / num_indices
	
	if mean[0] > 1:
		mean = (mean[0] / 255., mean[1] / 255., mean[2] / 255.)

	print('Mean is : ' + str(mean))

	return mean


def load_img_alpha(file_path):
	img = Image.open(file_path)
	if img.mode == 'RGBA':
		img.load()
		background = Image.new("RGB", img.size, (255,255,255))
		background.paste(img, mask=img.split()[3])
		img = background
	return img



def load_bw_img(file_path):
	img = Image.open(file_path)
	if img.mode == 'RGBA':
		img.load()
		img = img.split()[3]
	if img.mode == 'RGB':
		img.load()
		img = img.convert('1').split()[0].convert('L')
	if img.mode == 'L':
		img.load()
		img = img.convert('1').split()[0].convert('L')
	if img.mode == '1':
		img.load()
		img = img.split()[0].convert('L')
	if img.mode == 'P':
		img.load()
		img = img.convert('L')
	return img


import scipy
import scipy.io


def normalize(points):
	assert(points.size(1) == 3)
	min_v = points.min(dim=0, keepdim=True)[0]
	max_v = points.max(dim=0, keepdim=True)[0]
	
	# Move to center:
	points = points - (min_v + max_v) / 2.

	# Max offset:
	dimension = (max_v - min_v).max()
	dimension = 1. / dimension

	points = points * dimension
	return points


def normalize_np(points):
	assert(points.shape[1] == 3)
	min_v = points.min(axis=0, keepdims=True)
	max_v = points.max(axis=0, keepdims=True)

	# Move to center:
	points = points - (min_v + max_v) / 2.

	# Max offset:
	dimension = (max_v - min_v).max()
	dimension = 1. / dimension


	points = points * dimension
	return points

CHUNK_SIZE = 150
lenght_line = 60
import struct
import random
def get_n_random_lines_bin(path, n=5):
	with open(path, 'rb') as file:
		file.seek(0)
		chunk = file.read(4)

		file.seek(0, 2)
		size = file.tell()

		size = struct.unpack("<i", chunk)[0]

		file.seek(4)
		chunk = file.read(size*3*4)

		dt = np.dtype(np.float32)
		dt = dt.newbyteorder('<')
		pts = np.frombuffer(chunk, dtype=dt)

		pts = pts.reshape(-1,3)

		pts = normalize_np(pts)

		pts = pts[np.random.randint(0, pts.shape[0], n), :]

		return pts

def get_n_random_lines(path, n=5):
    MY_CHUNK_SIZE = lenght_line * (n+2)
    lenght = os.stat(path).st_size
    with open(path, 'r') as file:
            file.seek(random.randint(400, lenght - MY_CHUNK_SIZE))
            chunk = file.read(MY_CHUNK_SIZE)
            lines = chunk.split(os.linesep)
            return lines[1:n+1]

class ShapeNetCorePoints(data.Dataset):
	
	def getpointsvoxels(self, filenamepoints, filenamevoxels):
		with open(filenamepoints) as fp:
		    for i, line in enumerate(fp):
		        if i == 2:
		            try:
		                lenght = int(line.split()[2])
		            except ValueError:
		                print(filenamepoints)
		                print(line)
		            break

		for i in range(15):
			try:
			    mystring = get_n_random_lines(filenamepoints, n = 1024)
			    ptnormals = np.loadtxt(mystring).astype(np.float32)
			    break
			except ValueError as excep:
			    print(filenamepoints)
			    print(excep)

		binvox = ptnormals[:,0:3]
		binvox = torch.from_numpy(binvox).contiguous()
		binvox = normalize(binvox)

		voxels = torch.Tensor(scipy.io.loadmat(filenamevoxels)['voxel'])
	
		return binvox, voxels


class ShapeNet(data.Dataset):
	def __init__(self, scale=256, num_views=1, dataset_id=1, input_transform=None, seg_transform=None, random=0, mean=None, 
		directory=NIPS_LOCATION, class_id='03001627', max_theta=24):
		super(ShapeNet, self).__init__()

		self.dataset_id = dataset_id
		self.scale = scale

		self.num_views = num_views
		self.rng = np.random.RandomState(random)
		self.seg_transform = seg_transform

		if dataset_id == 1:
			folder_name = 'train'
		elif dataset_id == 2:
			folder_name = 'val'
		elif dataset_id >= 3:
			folder_name = 'test'

		self.max_theta = max_theta
		values = np.loadtxt(directory + 'shapenetcore_ids/%s_%sids.txt' % (class_id, folder_name), dtype='S200')
		self.images = [directory + 'shapenetcore_viewdata/%s/' % values[i] for i in range(0, values.shape[0])]
		self.mean = None

		self.input_transform = input_transform
		self.input_segtransform = input_transform(mean=(0,0,0), scale=scale)

		if mean == None:
			self.mean = get_avg_images(self.images, '', '/imgs/a015_e030.jpg', 100)
			self.input_transform = input_transform(mean=self.mean, scale=scale)
		else:
			self.mean = mean
			self.input_transform = input_transform(mean=self.mean, scale=scale)

	def __get_filename__(self, index):
		if self.dataset_id == 3:
			return self.images[int(index / 24 / 24)]
		elif self.dataset_id == 4:
			return self.images[index]
		return self.images[index]

	def __get_atlas_img__(self, index):
		img = self.images[index]
		view = (index % 24 + 1) * 15

		img_to_load = img + '/imgs/a%03d_e030.jpg' % view
		img_to_load = Image.open(img_to_load).convert("RGB")
		transforms = Compose([
							 Resize((256,256)),
		                     CenterCrop((224, 224)),
		                     ToTensor(),
		                     # normalize,
                        ])

		return transforms(img_to_load)[0:3,:,:]


	def __get_img_voxels__(self, index, batchsize=6, views=24):
		imgname = self.__get_filename__(index)

		filename = ('/'.join(imgname.split('/')[-3:]))[:-1]
		images = np.loadtxt('/scratch/local/ssd/ow/nips16_PTN/results_voxels/voxels_names.txt', dtype='S200')
		voxel_index = np.where(filename == images)[0][0]

		id = (voxel_index / 6) + 1
		voxel_list = np.load('/scratch/local/ssd/ow/nips16_PTN/results_voxels/voxels_%d.npy' % id)
		offset_voxel = voxel_index % 6
		final_index = offset_voxel * 24 + (index % 24)

		voxels = voxel_list[final_index,:,:,:]
		return torch.Tensor(voxels)

	def __len__(self):
		if self.dataset_id == 3:
			print('len', len(self.images * 24))
			return len(self.images) * 24 * 24
		elif self.dataset_id == 4:
			return len(self.images)
		elif self.dataset_id == 5:
			return len(self.images) * 24
		return len(self.images)

	def __getitem__(self, index):
		if self.dataset_id == 3:
			self.rng = np.random.RandomState(index)
			img_index1 = index % 24 + 1
			img_index2 = (index / 24) % 24 + 1
			views = [img_index1 * 15, img_index2 * 15] 
			for v in range(1, self.num_views):
				views = views+ [15 * self.rng.choice(range(1,25))]
			index = int(index / 24 / 24)
		elif self.dataset_id == 4:
			self.rng = np.random.RandomState(index)
			img_index1 = index % 24 + 1
			views = [img_index1 * 15]
			additional_views = (np.linspace(img_index1-1, img_index1-1+24, self.num_views+1) % 24 + 1) * 15
			additional_views = additional_views.astype(np.int32)
			additional_views = additional_views[:-1] 
			views = views + list(additional_views)
		elif self.dataset_id == 5:
			self.rng = np.random.RandomState(index)
			img_index1 = index % 24 + 1
			views = [img_index1 * 15] 
			for v in range(0, self.num_views):
				views = views+ [15 * self.rng.choice(range(1,self.max_theta+1))]
			index = int(index / 24)

		else:
			if self.num_views + 1 > self.max_theta + 1:
				views = [15 * d for d in self.rng.choice(range(1,self.max_theta+1), self.num_views+1, replace=True)]
			else:
				views = [15 * d for d in self.rng.choice(range(1,self.max_theta+1), self.num_views+1, replace=False)]

		seg_name = self.images[index] + '/masks/a%03d_e030.jpg' % views[0]
		target_name = self.images[index] + '/imgs/a%03d_e030.jpg' % views[0]
		seg_theta = torch.Tensor(get_sin_cos(np.pi / 180. * views[0]))


		input_seg = load_any_img(seg_name)
		target_img = self.input_transform(load_any_img(target_name))
		target_orig = self.seg_transform(load_any_img(target_name))
		views = views[1:]

		img_view_name = [0] * len(views)
		seg_view_name = [0] * len(views)

		img_views_no_transformation = [0] * len(views)
		img_views = [0] * len(views)
		depths = [0] * len(views)
		theta_views = [0] * len(views)
		seg_views = [0] * len(views)
		seg_subsample_views = [0] * len(views)
		
		subsample_masks = [0] * len(views)
		for i in range(0, len(views)):
			img_view_name[i] = self.images[index] + '/imgs/a%03d_e030.jpg' % views[i]
			seg_view_name[i] = self.images[index] + '/masks/a%03d_e030.jpg' % views[i]
			img_view = load_img_alpha(img_view_name[i])
			seg_view = load_bw_img(seg_view_name[i])
			if self.input_transform:
				img_view_no_transformation = self.seg_transform(img_view)
				img_view = self.input_transform(img_view) 
				
				img_views[i] = img_view
				seg_views[i] = (self.seg_transform(seg_view).float() > 0.5).float()
				img_views_no_transformation[i] = img_view_no_transformation
			theta_views[i] = torch.Tensor(get_sin_cos(np.pi / 180. * views[i]))
			seg_subsample_views[i] = self.input_segtransform(seg_view)
			subsample_masks[i] = seg_subsample_views[i] * 0 + 1
			
			depth_file = self.images[index].replace('shapenet_nips', 'shapenet_nips/depth') + '/depth_maps.npz'
			if not os.path.exists(depth_file):
				depths[i] = torch.zeros((1,256,256))
			else:
				depths[i] = torch.Tensor(np.load(self.images[index].replace('shapenet_nips', 'shapenet_nips/depth') + '/depth_maps.npz')['depthmaps'][views[i] / 15-1,:,:]).unsqueeze(0)

			if self.use_depth_silhouette:
				seg_views[i] = (depths[i] > 0).float()

				depths[i][seg_views[i] < 0.5] = 5
				depths[i] = depths[i] - 5 # Remap to between -1 -- 1

		if self.dataset_id == 3 or self.dataset_id == 5:
			input_seg = torch.Tensor(np.array(input_seg.resize((32,32), Image.ANTIALIAS))).float().unsqueeze(0) / 255.

		else:
			input_seg = (self.seg_transform(input_seg).float() > 0.5).float()

		return [img_views, seg_views, theta_views, target_img, input_seg, seg_theta.unsqueeze(2), \
				[], depths, seg_subsample_views, img_views_no_transformation + [target_orig]]
		

class DatasetFromMatFileSculptures(data.Dataset):
	def __init__(self, mat_file, use_mask=False, input_scale=256, output_scale=256,
		num_views=3, dataset_id=1, theta_transform=None,
		input_transform=None, seg_transform=None, random=0, mean=None, min_dTheta=-1, 
		base_filename=['oc_dataset_120/'], ending='.jpg'):
		"""
		Instantiates the dataset. Note that the index specifies whether is test/val/train
		"""
		super(DatasetFromMatFileSculptures, self).__init__()
		self.dataset_id = dataset_id
		print(mat_file)
		mat = scipy.io.loadmat(mat_file)

		self.num_views = num_views
		self.rng = np.random.RandomState(random)
		self.use_mask = use_mask
		self.random_view_selection = False
		self.seg_transform = seg_transform(scale=output_scale)
		self.input_segtransform = seg_transform(scale=input_scale)
		self.input_scale = input_scale
		self.output_scale = output_scale
		self.theta_transform = theta_transform

		self.all_base_filenames = ['/scratch/local/ssd/ow/%s/depth/' % b for b in base_filename]

		self.ending = ending

		self._read_matfile(mat, dataset_id, min_dTheta)
		self.idx = 0
		print("MEAN", mean)
		if mean == None:
			self.mean = get_avg_images(self.images['folder'][0][:], self.all_base_filenames[0], '/1.png', 100)
			print("NEW MEAN", self.mean)
			self.input_transform = input_transform(mean=self.mean, scale=input_scale)
		else:
			self.mean = mean
			self.input_transform = input_transform(mean=self.mean, scale=input_scale)


	def _read_matfile(self, mat_file, dataset_id, min_dTheta=-1):
		self.images = mat_file['imdb']['images'][0][0][0]

		print(sum(self.images['set'][0] == 1))
		print(sum(self.images['set'][0] == 2))
		print(sum(self.images['set'][0] == 3))
		print(self.all_base_filenames)
		if len(dataset_id) == 1:
			set_bool = self.images['set'][0] == dataset_id
		else:
			set_bool = (self.images['set'][0] == dataset_id[0]) | (self.images['set'][0] == dataset_id[1])


		elements = self.images.dtype.names
		self.num_elements = np.sum(set_bool)
		print(self.num_elements)

		# Select only the relevant values
		for i in range(0, len(elements)):
			if self.images[elements[i]][0].shape[0] == 1:
				self.images[elements[i]][0] = self.images[elements[i]][0][0][np.ravel(set_bool)]
			elif self.images[elements[i]][0].shape[1] > 1:
				self.images[elements[i]][0] = self.images[elements[i]][0][np.ravel(set_bool)]
			else:
				self.images[elements[i]][0] = self.images[elements[i]][0][set_bool]

		
		self.scale = 1.0

	def __len__(self):
		return self.num_elements * len(self.all_base_filenames)

	def getpoints(self, index):
		base_filename = self.all_base_filenames[int(index / (self.num_elements))]
		index = int(index / len(self.all_base_filenames))
		filename = BASE_FILE_TO_DIR + str(base_filename) + str(self.images['folder'][0][index][0]) + '/' + str(self.images['img_to_get'][0][index]) + self.ending
		pointfile = '/'.join(filename.replace('depth', '').split('/')[:-1])
		pointfile = [pointfile + '/' + f for f in os.listdir(pointfile) if f[-7:] == '.points'][0]
		for i in range(15):
			try:
			    ptnormals = get_n_random_lines_bin(pointfile, n = 1024)
			    break
			except ValueError as excep:
			    print(pointfile)
			    print(excep)

		binvox = ptnormals[:,0:3]
		binvox = torch.from_numpy(binvox).contiguous()

		return binvox

	def __get_atlas_img__(self, index):
		base_filename = self.all_base_filenames[int(index / (self.num_elements))]
		index = int(index / len(self.all_base_filenames))
		views = [self.views[0]]
		filename = BASE_FILE_TO_DIR + str(base_filename) + str(self.images['folder'][0][index][0]) + '/' + str(self.images['views'][0][index, views[0]]) + '.png'


		self.validating = Compose([
						Resize(137),
		                CenterCrop(127),
		                ])

		self.transforms = Compose([
							 Resize((256,256)),
		                     CenterCrop(224),
		                     ToTensor(),
		                     # normalize,
                        ])
		return self.transforms(self.validating(load_img(filename)))


	def __getitem__(self, index):
		self.base_filename = self.all_base_filenames[int(index / (self.num_elements))]

		index = int(index / len(self.all_base_filenames))

		FAILED_LOAD = False

		if not os.path.exists(BASE_FILE_TO_DIR + str(self.base_filename) + str(self.images['folder'][0][index][0]) + '/' + str(self.images['img_to_get'][0][index]) + self.ending):
			print("WARNING:: Missing %s" % (BASE_FILE_TO_DIR + str(self.base_filename) + str(self.images['folder'][0][index][0]) + '/' + str(self.images['img_to_get'][0][index]) + self.ending) 
		)
			seg_view = torch.zeros(1,self.output_scale,self.output_scale)
			input_seg = torch.zeros(1,self.input_scale,self.input_scale)
			index = 0
			input_img = torch.zeros(3,self.input_scale,self.input_scale)
			seg_theta = torch.Tensor(get_sin_cos(torch.Tensor(np.array([self.images['theta_to_get'][0][index]]))))
		
			FAILED_LOAD = True

		else:

			seg_name = BASE_FILE_TO_DIR + str(self.base_filename) + str(self.images['folder'][0][index][0]) + '/' + str(self.images['img_to_get'][0][index]) + '.exr.tiff'
			target_name = BASE_FILE_TO_DIR + str(self.base_filename) + str(self.images['folder'][0][index][0]) + '/' + str(self.images['img_to_get'][0][index]) + self.ending
			seg_theta = torch.Tensor(get_sin_cos((np.array([self.images['theta_to_get'][0][index]]))))
			input_seg = load_any_img(seg_name)
			input_img = load_img(target_name)
			input_img = self.input_transform(input_img)

			input_seg = (torch.Tensor(np.array(input_seg.resize((self.output_scale,self.output_scale), Image.NEAREST))) < 10).float().unsqueeze(0)
			input_seg_subsample = input_seg
			
		if self.random_view_selection:
			num_views = self.rng.randint(0,3)
			views = self.rng.choice([0,1,2], num_views+1,replace=False)
		elif hasattr(self, 'views'):
			views = self.views
		else:
			views = self.rng.choice([0,1,2], self.num_views, replace=False)
		imgs = ''

		img_view_name = [0] * len(views)
		seg_view_name = [0] * len(views)

		img_views_no_transformation = [0] * len(views)
		img_views = [0] * len(views)
		theta_views = [0] * len(views)
		depth_views = [0] * len(views)
		seg_views = [0] * len(views)
		seg_subsample_views = [0] * len(views)
		subsample_masks = [0] * len(views)

		for i in range(0, len(views)):
			img_view_name[i] = BASE_FILE_TO_DIR + str(self.base_filename + self.images['folder'][0][index][0]) + '/' + str(self.images['views'][0][index, views[i]]) + '.png'
			seg_view_name[i] = BASE_FILE_TO_DIR + str(self.base_filename + self.images['folder'][0][index][0]) + '/' + str(self.images['views'][0][index, views[i]]) + '.exr.tiff'
			depth_views[i] = BASE_FILE_TO_DIR + str(self.base_filename + self.images['folder'][0][index][0]) + '/' + str(self.images['views'][0][index, views[i]]) + '.exr.tiff'
			theta_views[i] = self.images['theta_views'][0][index, views[i]]
			
			if not os.path.exists(img_view_name[i]):
				print("WARNING: Input image doesn't exist")
				img_views[i] = torch.zeros(3,256,256)
				seg_views[i] = torch.zeros(1,256,256)
				depth_views[i] = torch.zeros(1,256,256)
				seg_subsample_views[i] = torch.zeros(3,256,256)
				img_views_no_transformation[i] = torch.zeros(3,256,256)
				FAILED_LOAD = True


			else:
				img_view = load_img(img_view_name[i])
				seg_view = load_any_img(seg_view_name[i])
				depth_view = load_any_img(depth_views[i])
				# if self.input_transform and self.dataset_id == 1:
				# 	img_view_no_transformation = self.seg_transform(img_view)
				# 	img_view = _jitter_location1(self.input_transform(img_view))
					
				# 	img_views[i] = img_view
				# 	seg_views[i] = 1 - _jitter_location1(self.seg_transform(seg_view).float())
				# 	img_views_no_transformation[i] = img_view_no_transformation
				# 	seg_subsample_views[i] = 1 - _jitter_location1(self.input_segtransform(seg_view).float())

				if self.input_transform:
					img_view_no_transformation = self.input_segtransform(img_view)
					img_view = self.input_transform(img_view) 
					
					img_views[i] = img_view
					depth_views[i] = torch.Tensor(np.array(depth_view.resize((self.output_scale,self.output_scale), Image.NEAREST))).unsqueeze(0)
					depth_views[i][depth_views[i] > 10] = 0
					seg_views[i] = (depth_views[i] > 0).float()

					depth_views[i] = torch.Tensor(np.array(depth_view.resize((self.input_scale,self.input_scale), Image.NEAREST))).unsqueeze(0)
					depth_views[i][depth_views[i] > 10] = 0
					subsample_masks[i] = (depth_views[i] > 0).float()
					
					#img_views[i] = (seg_views[i]).expand_as(img_views[i])
					img_views_no_transformation[i] = img_view_no_transformation
					seg_subsample_views[i] = seg_views[i]
			theta_views[i] = torch.Tensor(get_sin_cos(theta_views[i]))

		# if self.dataset_id == 1:
		# 	input_img = _jitter_location2(input_img)
		# 	input_seg = _jitter_location2(input_seg)

		return [img_views, seg_views, theta_views, input_img, input_seg, seg_theta, seg_subsample_views, depth_views, subsample_masks, img_views_no_transformation]

class DatasetFromTxtFileSculptures(data.Dataset):
	def __init__(self, txt_file, use_mask=False, input_scale=256, output_scale=256,
		num_views=3, dataset_id=1, theta_transform=None,
		input_transform=None, seg_transform=None, random=0, mean=None, test_augmentation=False, 
		):
		"""
		Instantiates the dataset. Note that the index specifies whether is test/val/train
		"""
		super(DatasetFromTxtFileSculptures, self).__init__()
		self.dataset_id = dataset_id
		self.test_augmentation = test_augmentation
		assert(len(dataset_id) == 1)

		self.images = np.load(txt_file[0] + '/angles_reshaped.npy').astype('|S200')
        # need to print
        print(np.load(txt_file[0] + '/angles_reshaped.npy'))
		# print(self.images.shape)		
		for i in range(1, len(txt_file)):
			self.images = np.vstack((self.images, np.load(txt_file[i] + '/angles_reshaped.npy').astype('|S200')))

		print(self.images.shape)
		print(self.images[0,:])
		self.images = self.images[self.images[:,-1].astype(np.float32).round() == dataset_id[0], :]
		print(self.images.shape)
		for i in range(0, self.images.shape[0]):
			self.images[i,-2] = self.images[i,-2].replace('augmented_additionalsculptures', txt_file[0] + '/augment_additionalsculptures')
			self.images[i,-2] = self.images[i,-2].replace('newtextures_oldsculptures', txt_file[0] + '/newtextures_oldsculptures/depth/')
			self.images[i,-2] = self.images[i,-2].replace('img_dataset_120c_fixedangle', txt_file[0] + '/img_dataset_120c_fixedangle/depth/')
		
		print(self.images.shape)
		t = self.images[:,0]
		for i in range(0, t.shape[0]):
			t[i] = str(self.images[i,-2]) + '/' + str(t[i])

		self.num_views = num_views
		self.rng = np.random.RandomState(random)
		self.use_mask = use_mask
		self.random_view_selection = False
		self.seg_transform = seg_transform(scale=output_scale)
		self.input_segtransform = seg_transform(scale=input_scale)
		self.input_scale = input_scale
		self.output_scale = output_scale
		self.theta_transform = theta_transform
		
		self.idx = 0
		print("MEAN", mean, self.num_views)
		if mean == None:

			print(t[0], BASE_FILE_TO_DIR)

			self.mean = get_avg_images(t, BASE_FILE_TO_DIR, '/1.png', 100)
			print("NEW MEAN", self.mean)
			self.input_transform = input_transform(mean=self.mean, scale=input_scale)
		else:
			self.mean = mean
			self.input_transform = input_transform(mean=self.mean, scale=input_scale)

	def __len__(self):
		return self.images.shape[0] - 1

	def __getitem__(self, orig_index):
		if self.dataset_id == 3 or self.dataset_id == 2:
			self.rng = np.random.RandomState(orig_index)

		FAILED_LOAD = False

		if self.test_augmentation:
			orig_index = int(orig_index / 20)
			angle_index = orig_index * 20
			index = orig_index * 20 + self.rng.randint(20)
		else:
			index = orig_index
			angle_index = int(orig_index / 20) * 20

		if hasattr(self, 'views'):
			views = [v + 1 for v in self.views]
		else:
			views = self.rng.choice([1,2,3,4,5], 4, replace=False)
			views = views[0:(self.num_views+1)]

		if not os.path.exists(BASE_FILE_TO_DIR + str(self.images[index, 0]) + '/' + str(views[0]) + '.png'):
			print("WARNING:: Missing %s" % (BASE_FILE_TO_DIR + str(self.images[index, 0]) + '/' + str(views[0]) + '.png')
		)
			seg_view = torch.zeros(1,self.output_scale,self.output_scale)
			input_seg = torch.zeros(1,self.input_scale,self.input_scale)
			index = 0
			input_img = torch.zeros(3,self.input_scale,self.input_scale)
			seg_theta = torch.Tensor(get_sin_cos(torch.Tensor(np.array([self.images[index, views[0]].astype(np.float32)]))))
		
			FAILED_LOAD = True

		else:

			seg_name = BASE_FILE_TO_DIR + str(self.images[index, 0]) + '/' + str(views[0]) + '.exr.tiff'
			target_name = BASE_FILE_TO_DIR + str(self.images[index, 0]) + '/' + str(views[0]) + '.png'
			seg_theta = torch.Tensor(get_sin_cos((np.array([self.images[angle_index, views[0]].astype(np.float32)]))))
			input_seg = load_any_img(seg_name)
			input_img = load_img(target_name)
			input_img = self.input_transform(input_img)

			try:
				input_seg = (torch.Tensor(np.array(input_seg.resize((self.output_scale,self.output_scale), Image.NEAREST))) < 10).float().unsqueeze(0)
			except:
				print(seg_name)
				print('FAILED')
				print(1+seg_name)

			input_seg_subsample = input_seg
			
		imgs = ''

		views = views[1:]

		img_view_name = [0] * len(views)
		seg_view_name = [0] * len(views)

		img_views_no_transformation = [0] * len(views)
		img_views = [0] * len(views)
		theta_views = [0] * len(views)
		depth_views = [0] * len(views)
		seg_views = [0] * len(views)
		seg_subsample_views = [0] * len(views)
		subsample_masks = [0] * len(views)
		for i in range(0, len(views)):
			if self.test_augmentation:
				index = orig_index*20 + self.rng.randint(20)
			img_view_name[i] = BASE_FILE_TO_DIR + str(self.images[index, 0]) + '/' + str(views[i]) + '.png'
			seg_view_name[i] = BASE_FILE_TO_DIR + str(self.images[index, 0]) + '/' + str(views[i]) + '.exr.tiff'
			depth_views[i] =   BASE_FILE_TO_DIR + str(self.images[index, 0]) + '/' + str(views[i]) + '.exr.tiff'
			theta_views[i] = self.images[angle_index, views[i]].astype(np.float32)
			if not os.path.exists(img_view_name[i]):
				print("WARNING: Input image doesn't exist", img_view_name[i])
				img_views[i] = torch.zeros(3,256,256)
				seg_views[i] = torch.zeros(1,256,256)
				depth_views[i] = torch.zeros(1,256,256)
				seg_subsample_views[i] = torch.zeros(3,256,256)
				img_views_no_transformation[i] = torch.zeros(3,256,256)
				FAILED_LOAD = True


			else:
				img_view = load_img(img_view_name[i])
				seg_view = load_any_img(seg_view_name[i])
				depth_view = load_any_img(depth_views[i])
				if self.input_transform:
					img_view_no_transformation = self.input_segtransform(img_view)
					img_view = self.input_transform(img_view) 
					
					img_views[i] = img_view
					try:
						depth_views[i] = torch.Tensor(np.array(depth_view.resize((self.output_scale,self.output_scale), Image.NEAREST))).unsqueeze(0)
					except:
						print(img_view_name[i])
						print('FAILED')
						print(1+img_view_name[i])

					depth_views[i][depth_views[i] > 10] = 0
					seg_views[i] = (depth_views[i] > 0).float()

					depth_views[i] = torch.Tensor(np.array(depth_view.resize((self.input_scale,self.input_scale), Image.NEAREST))).unsqueeze(0)
					depth_views[i][depth_views[i] > 10] = 0
					subsample_masks[i] = (depth_views[i] > 0).float()
					
					#img_views[i] = (seg_views[i]).expand_as(img_views[i])
					img_views_no_transformation[i] = img_view_no_transformation
					seg_subsample_views[i] = seg_views[i]
			theta_views[i] = torch.Tensor(get_sin_cos(theta_views[i]))

		return [img_views, seg_views, theta_views, input_img, input_seg, seg_theta, seg_subsample_views, depth_views, subsample_masks, img_views_no_transformation]
