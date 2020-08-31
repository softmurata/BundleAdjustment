
from dataset import *
from torchvision.transforms import Compose, Normalize, ToTensor, Scale, CenterCrop, Pad
import torchvision.transforms as transforms
import PIL
PADDING=0

def input_transform(mean=(83.4012 / 255., 81.1054 / 255., 78.6446 / 255.), scale=112, std=[1,1,1]):
    return Compose([
    	Scale(scale,interpolation=3),
    	Pad(padding=PADDING, fill=0),
    	Scale(scale,interpolation=3),
        ToTensor(),
        Normalize(mean=mean, std=std),
    ])


def input_transform_no_pad(mean=(83.4012 / 255., 81.1054 / 255., 78.6446 / 255.), scale=112, std=[1,1,1]):
    return Compose([
    	Scale(scale,interpolation=3),
        ToTensor(),
        Normalize(mean=mean, std=std),
    ])

def theta_transform():
	return Compose([
		ToTensor()
	])

def target_transform(scale=57):
    return Compose([
        Scale(scale, interpolation=3), 
    	Pad(padding=PADDING, fill=0),
    	Scale(scale,interpolation=3),
        ToTensor()
    ])

def target_transform_nopad(scale=57):
    return Compose([
        Scale(scale,interpolation=3),
        ToTensor()
    ])


def get_shapenet(dataset_id, num_views, scale, mean=None, random=0, input_scale=256, max_theta=24):
	print(scale, input_scale)
	return ShapeNet(dataset_id=dataset_id, 
		mean=mean,
		num_views=num_views,
		max_theta=max_theta,
		random=random,
		input_transform=input_transform_no_pad, scale=input_scale,
		seg_transform=target_transform_nopad(scale=(scale, scale)))

def get_sculpture_set(dataset_id, base_file, dTheta, random, num_views, scale, mean=None, min_dTheta=-1, input_scale=256):
	print("Dataset id", dataset_id, scale)
	if 1 in dataset_id:
        # vgg lab dataset format
        # /img_dataset_120c_fixedangle
     
		return DatasetFromTxtFileSculptures([base_file + '/img_dataset_120c_fixedangle/'], #base_file + '/augment_additionalsculptures/', base_file + '/img_dataset_120c_fixedangle/'], #[base_file + '/newtextures_oldsculptures/', base_file + '/augment_additionalsculptures/', base_file + '/img_dataset_120c_fixedangle/']
			dataset_id=dataset_id, output_scale=scale, input_scale=input_scale, 
			num_views=num_views,
			mean=mean,
			use_mask=True,
			theta_transform=theta_transform(), 
			input_transform=input_transform, 
			seg_transform=target_transform, random=random, test_augmentation=False)

	# elif 3 in dataset_id:
	# 	return DatasetFromTxtFileSculptures([base_file + '/img_dataset_120c_fixedangle/'], #[base_file + '/newtextures_oldsculptures/', base_file + '/augment_additionalsculptures/', base_file + '/img_dataset_120c_fixedangle/']
	# 		dataset_id=dataset_id, output_scale=scale,
	# 		num_views=num_views,
	# 		mean=mean,
	# 		use_mask=True,
	# 		theta_transform=theta_transform(), 
	# 		input_transform=input_transform, 
	# 		seg_transform=target_transform, random=random)

	elif 3 in dataset_id:
		return DatasetFromMatFileSculptures(base_file + '/oc/imdb/imdbviews3.mat',
			dataset_id=dataset_id, output_scale=scale, input_scale=input_scale,
			num_views=num_views,
			mean=mean,
			use_mask=True,
			theta_transform=theta_transform(), 
			input_transform=input_transform, 
			seg_transform=target_transform, random=random, min_dTheta=min_dTheta, base_filename=['img_dataset_120c_fixedangle'], ending='.png', 
			correspondences_filename='/scratch/local/ssd/ow/sculptures_dataset/correspondences/')
	elif 4 in dataset_id:
		return DatasetFromMatFileSculptures(base_file + '/sculptures_difftexture_samevp/', #base_file + '/oc/imdb/imdbviews3.mat',
			dataset_id=dataset_id, output_scale=scale,
			num_views=num_views,
			mean=mean,
			use_mask=True,
			theta_transform=theta_transform(), 
			input_transform=input_transform, 
			seg_transform=target_transform, random=random, min_dTheta=min_dTheta, base_filename=['img_dataset_120c_fixedangle'], ending='.png', 
			correspondences_filename='/scratch/local/ssd/ow/sculptures_dataset/correspondences/')

	else:
		return DatasetFromTxtFileSculptures([base_file + '/img_dataset_120c_fixedangle/'], #[base_file + '/new_textures_old_sculptures/'],
			dataset_id=dataset_id, output_scale=scale, input_scale=input_scale,
		#return DatasetFromTxtFileSculptures([base_file + '/sculptures_difftexture_samevp/'], #[base_file + '/img_dataset_120c_fixedangle/'],
			# dataset_id=dataset_id, output_scale=scale,
			num_views=num_views,
			mean=mean,
			use_mask=True,
			theta_transform=theta_transform(), 
			input_transform=input_transform, 
			seg_transform=target_transform, random=random, test_augmentation=False)
