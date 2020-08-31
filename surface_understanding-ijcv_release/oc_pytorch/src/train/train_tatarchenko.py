
from __future__ import print_function
import matplotlib
matplotlib.use('Agg') 
import sys
sys.path.append('./oc_pytorch/src/')
sys.path.append('/users/ow/relative-depth-using-pytorch/')
sys.path.append('./oc_pytorch/src/cyclic_consistency/')
import gc
import os
print(os.getcwd())
import shutil
import resource
import utils
from utils.conversions import copy_parameters, get_shared_parameters

import numpy as np
import argparse
from math import *
from data_utils.data import *
import data_utils.data as data
from data_utils.real_sculptures import RealSculptureDataset
from utils import network_utils

from Net.MVTatarchenko import Tatarchenko16

import torch
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from Net.SkipNet import Pix2PixModel
from experiments.visualise_gt_corrs import visualise_correspondences_forward_corrs
import model
hourglass = model.Best_model_period2_cpu
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn as nn

from utils.Globals import *


TRIMAP_SIZE = 1

# Training settings : a lot I know
parser = argparse.ArgumentParser(description='PyTorch encoder/3D decoder')
parser.add_argument('--batchSize', type=int, default=10, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=128, help='testing batch size')
parser.add_argument('--lr', type=float, default=0.00001, help='Learning Rate. Default=0.001')
parser.add_argument('--epoch', type=int, default=0, help='number of epochs to train for')
parser.add_argument('--continue_epoch', type=int, default=0, help='number of epochs to train for')
parser.add_argument('--num_views', type=int, default=1, help='number of epochs to train for')
parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('--cuda_device', type=int, help='cuda device')
parser.add_argument('--threads', type=int, default=10, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--imdb', type=str, default='/Users/oliviawiles/Documents/PhD/Database/oc/database/', help='location of dataset to use. Default=/data/ow/oc/imdb/')
parser.add_argument('--dataset', type=str, default='sculpture', help='database to use. Default=obj')
parser.add_argument('--model_epoch_path', type=str, default='/scratch/local/ssd/ow/code/runs/models/tatarchenko/', help='model epoch path')
parser.add_argument('--type_of_model', type=str, default='ImageBasedPrediction', help='type of model to be used')
parser.add_argument('--adam', type=bool, default=False)
parser.add_argument('--suffix', type=str, default='')


# Paramters about weight settings
parser.add_argument('--copy_weights', type=bool, default=False, help='whether to copy weights from another model')
parser.add_argument('--old_model', type=str, default='', help='location of the old model to use')
parser.add_argument('--test', default=False, action='store_true', help='location of the old model to use')


opt = parser.parse_args() 

writer = SummaryWriter('/scratch/local/ssd/ow/code/runs/tatarchenko/%s/' % opt.dataset)


ending =  "%s_size_%s" % (str(opt.lr), 'tatarchenko') + opt.suffix

imagenet_name = 'none'
WIDTH_OUTPUT=128

if opt.adam:
	opt.model_epoch_path = opt.model_epoch_path + '/%s/' % opt.dataset + opt.type_of_model + '_adam/'  + '_' + imagenet_name + '/num_views' + str(opt.num_views) + '_batchsize_' + str(opt.batchSize) 
else:
	opt.model_epoch_path = opt.model_epoch_path + '/%s/' % opt.dataset + opt.type_of_model + '_sgd/'   + '_' + imagenet_name + '/num_views' + str(opt.num_views) + '_batchsize_' + str(opt.batchSize) 


print('===> Building model')

model = Tatarchenko16()
model.mean = None
model = model.cuda()

torch.set_num_threads(opt.threads)

if opt.adam:
	optimizer = optim.Adam(list(model.parameters()), lr=opt.lr) # 0.00001 for silhouette 0.001 for other one ; 0.0001 when training silhouette_l1smooth
else:
	optimizer = optim.SGD(list(model.parameters()), lr=opt.lr, momentum=0.9) # 0.00001 for silhouette 0.001 for other one ; 0.0001 when training silhouette_l1smooth


print('====> Loading Datasets')
print(opt.dataset)

np.random.seed(opt.seed + opt.epoch)
torch.manual_seed(opt.seed + opt.epoch)


def get_dataset(epoch, dataset_id=3, percentage=1, num_to_use=-1):

	if opt.dataset == 'sculpture_quants':
		train_set = RealSculptureDataset(1, None, input_transform=data.input_transform_nn(mean=(0,0,0), scale=(256, 256)), \
		seg_transform=data.target_transform(scale=(256, 256)), theta_transform=data.theta_transform(), num_views=opt.num_views)
		test_set = RealSculptureDataset(dataset_id, None, input_transform=data.input_transform_nn(mean=(0,0,0), scale=(256, 256)), \
		seg_transform=data.target_transform(scale=(256, 256)), theta_transform=data.theta_transform(), num_views=opt.num_views)
	elif opt.dataset == 'sculpture':

	    train_set = get_sculpture_set([1], opt.imdb, False, random=epoch, num_views=opt.num_views, input_scale=WIDTH_OUTPUT, scale=WIDTH_OUTPUT,  mean=model.mean)
	    test_set = get_sculpture_set([dataset_id], opt.imdb, False, random=epoch, num_views=opt.num_views, input_scale=WIDTH_OUTPUT, scale=WIDTH_OUTPUT,  mean=train_set.mean)
	    if (model.mean == None):
	    	model.mean = train_set.mean
	elif opt.dataset == 'nips':
	    seg_size=WIDTH_OUTPUT
	    train_set = get_objnips_training_set(opt.imdb, False, random=opt.random_num_views, num_views=opt.num_views, dataset_id=1, mean=model.mean, seg_size=seg_size)
	    test_set = get_objnips_training_set(opt.imdb, False, random=opt.random_num_views, num_views=opt.num_views, dataset_id=2, mean=train_set.mean, seg_size=seg_size)
	    model.mean = train_set.mean
	elif opt.dataset == 'mvs':
		train_set = get_mvs_set(1, scale=WIDTH_OUTPUT, random=epoch, num_views=opt.num_views)
		test_set = get_mvs_set(2, scale=WIDTH_OUTPUT, random=epoch, num_views=opt.num_views)
	elif opt.dataset == 'sculptures_6k':
		if opt.use_large == 'hourglass':
			train_set = get_sculptures_6k_hourglass(1, scale=WIDTH_OUTPUT, random=epoch, num_views=opt.num_views)
			test_set = get_sculptures_6k_hourglass(dataset_id, scale=WIDTH_OUTPUT, random=epoch, num_views=opt.num_views)
		else:
			train_set = get_sculptures_6k(1, scale=WIDTH_OUTPUT, random=epoch, mean=model.mean, num_views=opt.num_views)
			train_set.percentage = percentage
			test_set = get_sculptures_6k(dataset_id, scale=WIDTH_OUTPUT, random=epoch, mean=train_set.mean, num_views=opt.num_views)
			if (model.mean == None):
				model.mean = train_set.mean
	elif opt.dataset == 'obj_120_3x':
		print('Using Imagenet?', opt.use_imagenet)
		print('Num views?', opt.num_views)
		if opt.use_imagenet:
			mean = (0.485, 0.456, 0.406)
		
			train_set = get_obj120_3x_set_imagenet(1, opt.imdb, False, random=epoch, num_views=opt.num_views, scale=WIDTH_OUTPUT,  mean=mean) #, mean=(83.4012 / 255., 81.1054 / 255., 78.6446 / 255.), min_dTheta=min_dTheta)
			test_set = get_obj120_3x_set_imagenet(dataset_id, opt.imdb, False, random=epoch, num_views=opt.num_views, scale=WIDTH_OUTPUT, mean=mean) #, mean=(83.4012 / 255., 81.1054 / 255., 78.6446 / 255.))
			if (model.mean == None):
				model.mean = train_set.mean
		else:
			train_set = get_obj120_3x_set(1, opt.imdb, False, random=epoch, num_views=opt.num_views, scale=WIDTH_OUTPUT,  mean=model.mean) #, mean=(83.4012 / 255., 81.1054 / 255., 78.6446 / 255.), min_dTheta=min_dTheta)
			test_set = get_obj120_3x_set(dataset_id, opt.imdb, False, random=epoch, num_views=opt.num_views, scale=WIDTH_OUTPUT, mean=train_set.mean) #, mean=(83.4012 / 255., 81.1054 / 255., 78.6446 / 255.))
			if (model.mean == None):
				model.mean = train_set.mean
	elif opt.dataset == 'horses_sil':
		train_set = get_horses_sil(1, scale=WIDTH_OUTPUT) #, mean=(83.4012 / 255., 81.1054 / 255., 78.6446 / 255.), min_dTheta=min_dTheta)
		test_set = get_horses_sil(2, scale=WIDTH_OUTPUT) #, mean=(83.4012 / 255., 81.1054 / 255., 78.6446 / 255.))
		if (model.mean == None):
			model.mean = train_set.mean
	elif opt.dataset == 'shapenet':
		train_set = get_shapenet(1, scale=WIDTH_OUTPUT, num_views=opt.num_views, random=epoch, mean=model.mean)
		test_set = get_shapenet(dataset_id, scale=WIDTH_OUTPUT, num_views=opt.num_views, random=epoch, mean=train_set.mean)
		if model.mean == None:
			model.mean = train_set.mean

		train_set.use_depth_silhouette = True
		test_set.use_depth_silhouette = True

	else:
	    train_set = get_obj_training_set(opt.imdb, False, random=epoch, num_views=opt.num_views, scale=WIDTH_OUTPUT)
	    test_set = get_obj_validation_set(opt.imdb, False, random=epoch, num_views=opt.num_views, scale=WIDTH_OUTPUT)

	if dataset_id == 3:
		test_set.views = range(0, opt.num_views)
	else:
		test_set.views = range(0,opt.num_views+1)

	training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
	testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)
	return training_data_loader, testing_data_loader

def get_batch_gt(batch, model, num_views=2, volatile=False, requires_grad=False, use_identity=False):
	inputs, input_segs, thetas, target_img, target, target_theta, seg_samples, gt_prediction, masks, orig_images = batch[0], batch[1], batch[2], batch[3], batch[4], batch[5], batch[6], batch[7], batch[8], batch[9]
	
	for i in range(0, len(gt_prediction)):
		gt_prediction[i] = Variable(gt_prediction[i], requires_grad=False, volatile=volatile)
		gt_prediction[i] = gt_prediction[i].cuda()

	for i in range(0, len(masks)):
		masks[i] = Variable(masks[i], requires_grad=False, volatile=volatile)
		masks[i] = masks[i].cuda()

	for i in range(0, len(inputs)):
		inputs[i] = Variable(inputs[i], requires_grad=requires_grad, volatile=volatile)
		thetas[i] = Variable(thetas[i], requires_grad=requires_grad, volatile=volatile)
		
		input_segs[i] = Variable(input_segs[i], requires_grad=requires_grad, volatile=volatile)

	for i in range(0, len(orig_images)):
		orig_images[i] = Variable(orig_images[i], requires_grad=False)
	target = Variable(target)
	target_theta = Variable(target_theta, requires_grad=requires_grad, volatile=volatile)
	target_img = Variable(target_img)

	if opt.cuda:
		for i in range(0, len(inputs)):
			inputs[i] = inputs[i].cuda()
			thetas[i] = thetas[i].cuda()
			input_segs[i] = input_segs[i].cuda()
		for i in range(0, len(orig_images)):
			orig_images[i] = orig_images[i].cuda()
		target_theta = target_theta.cuda()
		target = target.cuda()
		target_img = target_img.cuda()
	

	theta_input = thetas[0].squeeze()
	if len(theta_input.size()) == 1:
		theta_input = theta_input.unsqueeze(0)

	elevation = Variable(torch.zeros(theta_input.size()).cuda())
	elevation[:,0] = np.cos(0)
	elevation[:,1] = np.sin(0)

	theta_input = torch.cat((theta_input, elevation), 1)
	rgbd = model(target_img, theta_input)

	return (orig_images, inputs + [target_img]), masks, target, rgbd, gt_prediction, thetas, target_theta

def train_2views(epoch, num_views, percentage=1, num_to_use=1000):
	
	training_data_loader, testing_data_loader = get_dataset(epoch, percentage=percentage, dataset_id=1, num_to_use=num_to_use)
	epoch_loss = 0
	avg_loss1 = 0
	avg_loss2 = 0
	for iteration, batch in enumerate(training_data_loader, 1):
		(inputs, trans_inputs), input_segs, target, pred_depths, gt_depths, explainability, target_theta = get_batch_gt(batch, model, num_views=num_views, requires_grad=False)
		optimizer.zero_grad()

		gt_depths[0] = -nn.MaxPool2d(2)(-gt_depths[0])

		loss1 = nn.L1Loss()(pred_depths[:,3:,:,:], gt_depths[0])
		loss2 = nn.MSELoss()(pred_depths[:,:3,:,:], trans_inputs[0])
		#loss1 = loss1 / float(i+1)
		loss = loss1 * 0.1 + loss2
		avg_loss1 += loss1.cpu().data[0]
		avg_loss2 += loss2.cpu().data[0]

		epoch_loss += loss.data[0]
		loss.backward()

		optimizer.step()

		if (iteration % 500 == 0 and epoch == 0) or iteration == 1:
			x = vutils.make_grid(trans_inputs[0][0:10,:,:].data, normalize=True, scale_each=True)
			writer.add_image('Train/Inputs_%d' % iteration, x, epoch)
			x = vutils.make_grid(trans_inputs[-1][0:10,:,:].data, normalize=True, scale_each=True)
			writer.add_image('Train/Outputs_%d' % iteration, x, epoch)
			x = vutils.make_grid(gt_depths[0][0:10,:,:].data, normalize=True)
			writer.add_image('Train/GT_depth_%d' % iteration, x, epoch)

			x = vutils.make_grid(pred_depths[0:10,3:,:].data, normalize=True, scale_each=True)
			writer.add_image('Train/Pred_depth_%d' % iteration, x, epoch)
			x = vutils.make_grid(pred_depths[0:10,0:3,:].data, normalize=True, scale_each=True)
			writer.add_image('Train/Pred_rgb_%d' % iteration, x, epoch)
		

		loss_mean = epoch_loss / iteration

		if iteration % 10 == 0:
			print("===> Epoch[{}]({}/{}): Loss: {:.4f}; Avg loss 1: {:.4f}; Avg loss 2 {:.4f}".format(epoch, 
				iteration, len(training_data_loader), loss_mean, avg_loss1 / float(iteration), avg_loss2/ float(iteration)))

	print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(training_data_loader)))
	return loss_mean, avg_loss1, avg_loss2

def val(epoch, num_views, iter_vis=100, prefix=''):
	
	training_data_loader, testing_data_loader = get_dataset(epoch, dataset_id=2)
	epoch_loss = 0
	avg_loss1 = 0
	avg_loss2 = 0
	for iteration, batch in enumerate(testing_data_loader, 1):
		(inputs, trans_inputs), input_segs, target, pred_depths, gt_depths, explainability, target_theta = get_batch_gt(batch, model, num_views=num_views, requires_grad=False)
		gt_depths[0] = -nn.MaxPool2d(2)(-gt_depths[0])
		loss1 = nn.L1Loss()(pred_depths[:,3:,:,:],  gt_depths[0])
		loss2 = nn.MSELoss()(pred_depths[:,:3,:,:], trans_inputs[0])
		#loss1 = loss1 / float(i+1)
		loss = loss1 * 0.1 + loss2
		avg_loss1 += loss1.cpu().data[0]
		avg_loss2 += loss2.cpu().data[0]

		epoch_loss += loss.data[0]

		if (iteration % 1000 == 0 and epoch == 0) or iteration == 1:
			x = vutils.make_grid(trans_inputs[0][0:10,:,:].data, normalize=True, scale_each=True)
			writer.add_image('Val/Inputs_%d' % iteration, x, epoch)
			x = vutils.make_grid(trans_inputs[-1][0:10,:,:].data, normalize=True, scale_each=True)
			writer.add_image('Val/Outputs_%d' % iteration, x, epoch)
			x = vutils.make_grid(gt_depths[0][0:10,:,:].data, normalize=True)
			writer.add_image('Val/GT_depth_%d' % iteration, x, epoch)

			x = vutils.make_grid(pred_depths[0:10,3:,:].data, normalize=True, scale_each=True)
			writer.add_image('Val/Pred_depth_%d' % iteration, x, epoch)
			x = vutils.make_grid(pred_depths[0:10,0:3,:].data, normalize=True, scale_each=True)
			writer.add_image('Val/Pred_rgb_%d' % iteration, x, epoch)
		

		loss_mean = epoch_loss / iteration

		if iteration % 10 == 0:
			print("===> Epoch[{}]({}/{}): Loss: {:.4f}; Avg loss 1: {:.4f}; Avg loss 2 {:.4f}".format(epoch, 
				iteration, len(testing_data_loader), loss_mean, avg_loss1 / float(iteration), avg_loss2/ float(iteration)))

	print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(testing_data_loader)))
	return loss_mean, avg_loss1, avg_loss2

def checkpoint(epoch):
	dict = {'epoch' : epoch, 'state_dict' : model.state_dict(), 'optimizer' : optimizer.state_dict(), 'mean' : model.mean}
		
	model_out_path = "{}model_epoch_{}.pth".format(opt.model_epoch_path, epoch)
	print(os.path.exists(opt.model_epoch_path))
	if not(os.path.exists(opt.model_epoch_path)):
		os.makedirs(opt.model_epoch_path)
	torch.save(dict, model_out_path)
	print("Checkpoint saved to {}".format(model_out_path))

	# remove all previous ones
	for i in range(0, epoch-1):	
		if os.path.exists("{}model_epoch_{}.pth".format(opt.model_epoch_path, i)):
			os.remove( "{}model_epoch_{}.pth".format(opt.model_epoch_path, i))

import torch.optim.lr_scheduler as lr_scheduler
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10)

last_epoch = -1
best_epoch = 1000
if not(opt.test):
	percentage = 1
	for epoch in range(opt.continue_epoch,1000):
		
		if os.environ.get('CUDA_DEVICE'):
			torch.cuda.set_device(int(os.environ.get('CUDA_DEVICE')))
		else:
			torch.cuda.set_device(3)

		if opt.cuda and not torch.cuda.is_available():
			raise Exception("No GPU found, run without --cuda")
		torch.cuda.manual_seed(opt.seed)

		if epoch > 0:
			print(">>>>>>> Loading model epoch %d" % (epoch-1))
			print("{}model_epoch_{}.pth".format(opt.model_epoch_path, epoch))
			checkpoint_file = torch.load("{}model_epoch_{}.pth".format(opt.model_epoch_path, epoch-1))
			model.load_state_dict(checkpoint_file['state_dict'])
			
			optimizer.load_state_dict(checkpoint_file['optimizer'])
			print("Loaded checkpoint")

		model = model.cuda()

		model.train()
		
		tloss, tl1, tl2 = train_2views(epoch, opt.num_views, percentage, num_to_use=-1)
		with torch.no_grad():
			model.eval()
		
			vloss, vl1, vl2 = val(45, opt.num_views, iter_vis=50, prefix=opt.suffix)

		writer.add_scalars('loss/trainval', {'train' : tloss, 'val' : vloss}, epoch)
		writer.add_scalars('l1/trainval', {'train' : tl1, 'val' : vl1}, epoch)
		writer.add_scalars('l2/trainval', {'train' : tl2, 'val' : vl2}, epoch)
		
		checkpoint(epoch)
