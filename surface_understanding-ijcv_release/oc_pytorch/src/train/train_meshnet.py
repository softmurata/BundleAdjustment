
from __future__ import print_function
import matplotlib
matplotlib.use('Agg') 
import sys
sys.path.append('./oc_pytorch/src/')
sys.path.append('/users/ow/relative-depth-using-pytorch/')
sys.path.append('./oc_pytorch/src/cyclic_consistency/')
import gc
import os
import shutil
import resource
import utils

import numpy as np
import argparse
from math import *
from data_utils.data import *
import data_utils.data as data

from tensorboardX import SummaryWriter
import torch
from Net.SkipNet import Pix2PixModel
from Net.MeshBasedPrediction import SilhouettePrediction, AngleEncoder
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn as nn

from utils.Globals import *


TRIMAP_SIZE = 1

# Training settings : a lot I know
parser = argparse.ArgumentParser(description='PyTorch encoder/3D decoder')
parser.add_argument('--batchSize', type=int, default=10, help='training batch size')
parser.add_argument('--loss', type=str, default='l1', help='whether to use a random number of views')
parser.add_argument('--testBatchSize', type=int, default=128, help='testing batch size')
parser.add_argument('--lr', type=float, default=0.00001, help='Learning Rate. Default=0.001')
parser.add_argument('--epoch', type=int, default=0, help='number of epochs to train for')
parser.add_argument('--iters', type=int, default=0, help='number of epochs to train for')
parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('--cuda_device', type=int, help='cuda device')
parser.add_argument('--predict_silhouette', type=bool, default=False)
parser.add_argument('--threads', type=int, default=10, help='number of threads for data loader to use')
parser.add_argument('--continue_epoch', type=int, default=0, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--imdb', type=str, default='/Users/oliviawiles/Documents/PhD/Database/oc/database/', help='location of dataset to use. Default=/data/ow/oc/imdb/')
parser.add_argument('--dataset', type=str, default='sculpture', help='database to use. Default=obj')
parser.add_argument('--model_epoch_path', type=str, default='./', help='model epoch path')
parser.add_argument('--type_of_model', type=str, default='ImageBasedPrediction', help='type of model to be used')
parser.add_argument('--num_views', type=int, default=5, help='number of views to use')
parser.add_argument('--adam', type=bool, default=False)
parser.add_argument('--use_large', type=str, default='')
parser.add_argument('--use_imagenet', action='store_true')
parser.add_argument('--use_percentage', action='store_true')
parser.add_argument('--test_angles', action='store_true')
parser.add_argument('--use_trainval', action='store_true')
parser.add_argument('--use_hard', action='store_true')
parser.add_argument('--use_cleaned_dataset_curr', action='store_true')
parser.add_argument('--suffix', type=str, default='')
parser.add_argument('--max_theta', type=int, default=24)

# Paramters about weight settings
parser.add_argument('--copy_weights', type=bool, default=False, help='whether to copy weights from another model')
parser.add_argument('--old_model', type=str, default='', help='location of the old model to use')
parser.add_argument('--test', default=False, action='store_true', help='location of the old model to use')

parser.add_argument('--lambda1', type=float, default=10, help='location of the old model to use')
parser.add_argument('--lambda2', type=float, default=1, help='location of the old model to use')

opt = parser.parse_args() 

if SMALL_DECODER_3D:
	opt.suffix += '_small3d'

opt.suffix += '_nv%d_mt%d' % (opt.num_views, opt.max_theta)

if opt.max_theta != 24:
	opt.suffix += 'theta_' + str(opt.max_theta)

if opt.use_large == 'large':
	large_name = 'vgg'
elif opt.use_large == 'pix2pix':
	large_name = 'pix2pix'
elif opt.use_large == 'hourglass':
	large_name = 'hourglass'
else:
	large_name = 'silnet'

ending =  "%s_%s_size_%s" % (opt.loss, str(opt.lr), large_name) + opt.suffix

if opt.use_imagenet:
	imagenet_name = 'imagenet'
else:
	imagenet_name = 'none'

if opt.adam:
	opt.model_epoch_path = opt.model_epoch_path + '/%s/' % opt.dataset + opt.type_of_model + '_adam/' + large_name + '_' + imagenet_name + '/num_views' + str(opt.num_views) + '_batchsize_' + str(opt.batchSize) + 'finetune' + str(opt.copy_weights) + '_loss_' + opt.loss + '_lambda1_2' + str(opt.lambda1) + '_' + str(opt.lambda2) + ending
else:
	opt.model_epoch_path = opt.model_epoch_path + '/%s/' % opt.dataset + opt.type_of_model + '_sgd/'  + large_name + '_' + imagenet_name + '/num_views' + str(opt.num_views) + '_batchsize_' + str(opt.batchSize) + 'finetune' + str(opt.copy_weights) + '_loss_' + opt.loss + '_lambda1_2' + str(opt.lambda1) + '_' + str(opt.lambda2) + ending

writer = SummaryWriter('/scratch/local/ssd/ow/code/runs/sidenet/%s_%s/' % (opt.dataset, opt.suffix))

print('===> Building model')
print(opt.type_of_model)

if (SMALL_DECODER or SMALL_DECODER_3D) and not(opt.test):
	WIDTH_OUTPUT = 57
else:
	WIDTH_OUTPUT = 256


silhouette_prediction = SilhouettePrediction()
angle_encoder = AngleEncoder()

if opt.copy_weights:
	model = Pix2PixModel(loss_type=opt.loss, num_variables=16)
	state = model.state_dict()
	state.update(torch.load(opt.old_model)['state_dict'])
	model.load_state_dict(state)
	if POST_DEPTH:
		angle_encoder.load_state_dict(torch.load(opt.old_model)['angle_encoder'])
	
	state = silhouette_prediction.state_dict()
	state.update(torch.load(opt.old_model)['silhouette_state'])
	silhouette_prediction.load_state_dict(state)
else:
	model = Pix2PixModel(loss_type=opt.loss, num_variables=16)


model.stats = {'silhouette':np.zeros((0,1)), 'error':np.zeros((0,1)), 'trimap_3':np.zeros((0,5)), 'trimap_5':np.zeros((0,5)), 'loss':np.zeros((0,1))}
model.val_stats = {'silhouette':np.zeros((0,1)), 'error':np.zeros((0,1)), 'id_1':np.zeros((0,1)),'id_2':np.zeros((0,1)),'loss':np.zeros((0,1)), 
						'sil_iou_inside' : np.zeros((0,1)), 'sil_iou_outside' : np.zeros((0,1)),
						'vis12' : np.zeros((0,1)), 'invis12' : np.zeros((0,1)), 'vis21' : np.zeros((0,1)), 'invis21' : np.zeros((0,1))}
model.iterstats = {'accuracy':np.zeros((5,)),  'trimap_2':np.zeros((5,)), 'trimap_3':np.zeros((5,)), 'trimap_5':np.zeros((5,)),'loss':np.zeros((1,))}

model.iteration = 0


torch.set_num_threads(opt.threads)

silhouette_prediction = silhouette_prediction.cuda()
model = model.cuda()
if opt.adam:
	optimizer = optim.Adam(list(model.parameters()) + list(silhouette_prediction.parameters()), lr=opt.lr) # 0.00001 for silhouette 0.001 for other one ; 0.0001 when training silhouette_l1smooth
else:
	optimizer = optim.SGD(list(model.parameters()) + list(silhouette_prediction.parameters()), lr=opt.lr, momentum=0.9) # 0.00001 for silhouette 0.001 for other one ; 0.0001 when training silhouette_l1smooth


print('====> Loading Datasets')
print(opt.dataset)

np.random.seed(opt.seed + opt.epoch)
torch.manual_seed(opt.seed + opt.epoch)

if opt.copy_weights and not(opt.use_large == 'hourglass'):
	model.mean = torch.load(opt.old_model)['mean']
else:
	model.mean = None


def get_dataset(epoch, dataset_id=3, percentage=1, num_to_use=-1, use_cleaned_dataset=opt.use_cleaned_dataset_curr):

	if opt.dataset == 'sculpture':

	    train_set = get_sculpture_set([1], opt.imdb, False, random=epoch, num_views=opt.num_views, scale=WIDTH_OUTPUT,  mean=model.mean)
	    test_set = get_sculpture_set([dataset_id], opt.imdb, False, random=epoch, num_views=opt.num_views, scale=WIDTH_OUTPUT,  mean=train_set.mean)
	    if (model.mean == None):
	    	model.mean = train_set.mean
	elif opt.dataset == 'shapenet':
		train_set = get_shapenet(1, scale=WIDTH_OUTPUT, num_views=opt.num_views, random=epoch, mean=model.mean, max_theta=opt.max_theta)
		test_set = get_shapenet(dataset_id, scale=WIDTH_OUTPUT, num_views=opt.num_views, random=epoch, mean=train_set.mean, max_theta=opt.max_theta)
		if opt.lambda1 > 0:
			train_set.use_depth_silhouette = True
			test_set.use_depth_silhouette = True
		else:
			train_set.use_depth_silhouette = False
			test_set.use_depth_silhouette = False

		if model.mean == None:
			model.mean = train_set.mean

	test_set.loss = opt.loss
	train_set.loss = opt.loss

	if dataset_id == 3:
		test_set.views = range(0, opt.num_views+1)
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
	
	input_args = []
	for i in range(0, len(inputs)):
		input_args += [inputs[i], thetas[i].unsqueeze(3), angle_encoder(thetas[i].unsqueeze(3))]

	print(len(input_args), opt.num_views, len(inputs), len(orig_images))
	heights = model(*input_args)
	fv_silhouette = heights[-1]
	heights = heights[:-1]

	return (orig_images, inputs + [target_img]), masks, target, heights, gt_prediction, thetas, fv_silhouette, target_theta

def train_2views(epoch, num_views, percentage=1, num_to_use=1000):
	
	avg_num_corr = 0
	avg_image = torch.zeros(100,100)
	pix_divide = torch.zeros(100,100)
	training_data_loader, testing_data_loader = get_dataset(epoch, percentage=percentage, dataset_id=2, num_to_use=num_to_use)
	epoch_loss = 0
	avg_loss1 = 0
	avg_loss2 = 0
	for iteration, batch in enumerate(training_data_loader, 1):
		(inputs, trans_inputs), input_segs, target, pred_depths, gt_depths, explainability, feature_vector, target_theta = get_batch_gt(batch, model, num_views=num_views, requires_grad=False)
		
		pred_depths = list(pred_depths)
		gt_depths = list(gt_depths)

		if opt.loss == 'DepthSilLoss':

			pred_depths[0] = pred_depths[0] * (input_segs[0] > 0.5).float()
			gt_depths[0] = gt_depths[0] * (input_segs[0] > 0.5).float()

			if MEAN_DEPTH:
				mean_pred_depth = pred_depths[0].sum(dim=3).sum(dim=2).sum(dim=1) / (input_segs[0] > 0.5).float().sum(dim=3).sum(dim=2).sum(dim=1).clamp(min=1)
				mean_gt_depth = gt_depths[0].sum(dim=3).sum(dim=2).sum(dim=1) / (input_segs[0] > 0.5).float().sum(dim=3).sum(dim=2).sum(dim=1).clamp(min=1)
				
				pred_depths[0] = pred_depths[0] - Variable(mean_pred_depth.unsqueeze(1).unsqueeze(1).unsqueeze(1).data)
				gt_depths[0] = gt_depths[0] - Variable(mean_gt_depth.unsqueeze(1).unsqueeze(1).unsqueeze(1).data)

				pred_depths[0] = pred_depths[0] * (input_segs[0] > 0.5).float()
				gt_depths[0] = gt_depths[0] * (input_segs[0] > 0.5).float()

			loss1 = (nn.L1Loss(reduce=False).cuda()(pred_depths[0] , gt_depths[0]).sum(dim=3).sum(dim=2).sum(dim=1) /(input_segs[0] > 0.5).float().sum(dim=3).sum(dim=2).sum(dim=1).clamp(min=1)).mean()
			
			i = 0
			for i in range(1, len(pred_depths)):
				pred_depths[i] = pred_depths[i] * (input_segs[i] > 0.5).float()
				gt_depths[i] = gt_depths[i] * (input_segs[i] > 0.5).float()
				if MEAN_DEPTH:
					mean_pred_depth = pred_depths[i].sum(dim=3).sum(dim=2).sum(dim=1) / (input_segs[i] > 0.5).float().sum(dim=3).sum(dim=2).sum(dim=1).clamp(min=1)
					mean_gt_depth = gt_depths[i].sum(dim=3).sum(dim=2).sum(dim=1) / (input_segs[i] > 0.5).float().sum(dim=3).sum(dim=2).sum(dim=1).clamp(min=1)


					pred_depths[i] = pred_depths[i] - Variable(mean_pred_depth.unsqueeze(1).unsqueeze(1).unsqueeze(1).data)
					gt_depths[i] = gt_depths[i] - Variable(mean_gt_depth.unsqueeze(1).unsqueeze(1).unsqueeze(1).data)

					pred_depths[i] = pred_depths[i] * (input_segs[i] > 0.5).float()
					gt_depths[i] = gt_depths[i] * (input_segs[i] > 0.5).float()

				loss1 += (nn.L1Loss(reduce=False).cuda()(pred_depths[i] , gt_depths[i]).sum(dim=3).sum(dim=2).sum(dim=1) /(input_segs[i] > 0.5).float().sum(dim=3).sum(dim=2).sum(dim=1).clamp(min=1)).mean()


			#loss1 = loss1 / float(i+1)
			loss2, predicted_silhouette = silhouette_prediction(target_theta, feature_vector, target)
			loss = loss1 * opt.lambda1 + loss2 * opt.lambda2
			avg_loss1 += loss1.cpu().data[0]
			avg_loss2 += loss2.cpu().data[0]

		optimizer.zero_grad()
		epoch_loss += loss.data[0]
		loss.backward()

		optimizer.step()

		del explainability
		del feature_vector
		del target_theta
		del loss
		del loss1
		del loss2


		if iteration == 1 or iteration % 1000 == 0:

			for i in range(0,len(input_segs)):
				writer.add_image('Train/input%d_%d' % (i, iteration), inputs[i][0:10,:,:].data, epoch)
				writer.add_image('Train/inputseg%d_%d' % (i, iteration),input_segs[i][0:10,:,:].data, epoch)
				writer.add_image('Train/transinput%d_%d' % (i, iteration),trans_inputs[i][0:10,:,:].data, epoch)
				writer.add_image('Train/preddepth%d_%d' % (i, iteration),pred_depths[i][0:10,:,:].data, epoch)
				writer.add_image('Train/gtdepth%d_%d' % (i, iteration),gt_depths[i][0:10,:,:].data, epoch)

			writer.add_image('Train/predsil%d' % i, predicted_silhouette[0:10,:,:,:].data, epoch)
			writer.add_image('Train/target%d' % i, target[0:10,:,:,:].data,epoch)


		loss_mean = epoch_loss / iteration
		print("===> Epoch[{}]({}/{}): Loss: {:.4f}; Avg loss 1: {:.4f}; Avg loss 2 {:.4f}".format(epoch, 
			iteration, len(training_data_loader), loss_mean, avg_loss1 / float(iteration), avg_loss2/ float(iteration)))

		# Forceablly clean up : can't have the network twice!
		del inputs
		del trans_inputs
		del input_segs
		del pred_depths
		del gt_depths
		del target
		del predicted_silhouette

	model.stats['loss'] = np.vstack((model.stats['loss'], epoch_loss / iteration))
	model.stats['silhouette'] = np.vstack((model.stats['silhouette'], avg_loss2 / iteration))
	model.stats['error'] = np.vstack((model.stats['error'], avg_loss1 / iteration))
	print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(training_data_loader)))
	return loss_mean, avg_loss1 / float(iteration), avg_loss2 / float(iteration)

def val(epoch, num_views, iter_vis=100, prefix=''):
	
	val_loss = 0
	avg_loss1 = 0
	avg_loss2 = 0
	num = 0
	val_accuracy = np.zeros((5,))
	training_data_loader, testing_data_loader = get_dataset(epoch, dataset_id=2)
	for iteration, batch in enumerate(testing_data_loader, 1):
		(inputs, trans_inputs), input_segs, target, pred_depths, gt_depths, explainability, feature_vector, target_theta = get_batch_gt(batch, model, num_views=num_views, requires_grad=False, volatile=True)
		
		pred_depths = list(pred_depths)
		gt_depths = list(gt_depths)

		if opt.loss == 'DepthSilLoss':
			pred_depths[0] = pred_depths[0] * (input_segs[0] > 0.5).float()
			gt_depths[0] = gt_depths[0] * (input_segs[0] > 0.5).float()
			
			if MEAN_DEPTH:
				mean_pred_depth = pred_depths[0].sum(dim=3).sum(dim=2).sum(dim=1) / (input_segs[0] > 0.5).float().sum(dim=3).sum(dim=2).sum(dim=1).clamp(min=1)
				mean_gt_depth = gt_depths[0].sum(dim=3).sum(dim=2).sum(dim=1) / (input_segs[0] > 0.5).float().sum(dim=3).sum(dim=2).sum(dim=1).clamp(min=1)
				
				pred_depths[0] = pred_depths[0] - Variable(mean_pred_depth.unsqueeze(1).unsqueeze(1).unsqueeze(1).data)
				gt_depths[0] = gt_depths[0] - Variable(mean_gt_depth.unsqueeze(1).unsqueeze(1).unsqueeze(1).data)

				pred_depths[0] = pred_depths[0] * (input_segs[0] > 0.5).float()
				gt_depths[0] = gt_depths[0] * (input_segs[0] > 0.5).float()

			loss1 = (nn.L1Loss(reduce=False).cuda()(pred_depths[0] , gt_depths[0]).sum(dim=3).sum(dim=2).sum(dim=1) /(input_segs[0] > 0.5).float().sum(dim=3).sum(dim=2).sum(dim=1).clamp(min=1)).mean()
			
			i = 0
			for i in range(1, len(pred_depths)):
				pred_depths[i] = pred_depths[i] * (input_segs[i] > 0.5).float()
				gt_depths[i] = gt_depths[i] * (input_segs[i] > 0.5).float()
				if MEAN_DEPTH:
					mean_pred_depth = pred_depths[i].sum(dim=3).sum(dim=2).sum(dim=1) / (input_segs[i] > 0.5).float().sum(dim=3).sum(dim=2).sum(dim=1).clamp(min=1)
					mean_gt_depth = gt_depths[i].sum(dim=3).sum(dim=2).sum(dim=1) / (input_segs[i] > 0.5).float().sum(dim=3).sum(dim=2).sum(dim=1).clamp(min=1)


					pred_depths[i] = pred_depths[i] - Variable(mean_pred_depth.unsqueeze(1).unsqueeze(1).unsqueeze(1).data)
					gt_depths[i] = gt_depths[i] - Variable(mean_gt_depth.unsqueeze(1).unsqueeze(1).unsqueeze(1).data)

					pred_depths[i] = pred_depths[i] * (input_segs[i] > 0.5).float()
					gt_depths[i] = gt_depths[i] * (input_segs[i] > 0.5).float()

				loss1 += (nn.L1Loss(reduce=False).cuda()(pred_depths[i] , gt_depths[i]).sum(dim=3).sum(dim=2).sum(dim=1) /(input_segs[i] > 0.5).float().sum(dim=3).sum(dim=2).sum(dim=1).clamp(min=1)).mean()

			loss1 = loss1 / float(i+1)
			loss2, predicted_silhouette = silhouette_prediction(target_theta, feature_vector, target)
			loss = loss1 * opt.lambda1 + loss2 * opt.lambda2
			avg_loss1 += loss1.cpu().data[0]
			avg_loss2 += loss2.cpu().data[0]
			val_loss += loss.cpu().data[0]

		
		optimizer.zero_grad()

		del explainability
		del feature_vector
		del target_theta
		del loss
		del loss1
		del loss2


		if iteration == 1 or iteration % 1000 == 0:

			for i in range(0,len(input_segs)):
				writer.add_image('Val/input%d_%d' % (i, iteration), inputs[i][0:10,:,:].data, epoch)
				writer.add_image('Val/inputseg%d_%d' % (i, iteration),input_segs[i][0:10,:,:].data, epoch)
				writer.add_image('Val/transinput%d_%d' % (i, iteration),trans_inputs[i][0:10,:,:].data, epoch)
				writer.add_image('Val/preddepth%d_%d' % (i, iteration),pred_depths[i][0:10,:,:].data, epoch)
				writer.add_image('Val/gtdepth%d_%d' % (i, iteration),gt_depths[i][0:10,:,:].data, epoch)

			writer.add_image('Val/predsil%d' % i, predicted_silhouette[0:10,:,:,:].data, epoch)
			writer.add_image('Val/target%d' % i, target[0:10,:,:,:].data,epoch)

		num = num + input_segs[0].size(0)
		print("===> Epoch[{}]({}/{}): Loss: {:.4f}; Avg loss 1: {:.4f}; Avg loss 2 {:.4f}".format(epoch, 
			iteration, len(testing_data_loader), val_loss / float(iteration), avg_loss1 / float(iteration), avg_loss2 / float(iteration)))

		# Forceablly clean up : can't have the network twice!
		del inputs
		del trans_inputs
		del input_segs
		del pred_depths
		del gt_depths
		del target
		del predicted_silhouette
		
	model.val_stats['loss'] = np.vstack((model.val_stats['loss'], val_loss / float(iteration)))
	model.val_stats['silhouette'] = np.vstack((model.val_stats['silhouette'], avg_loss2 / float(iteration)))
	model.val_stats['error'] = np.vstack((model.val_stats['error'], avg_loss1 / float(iteration)))
	model.iteration = 0	
	return (val_loss / float(iteration)), (avg_loss1 / float(iteration)), (avg_loss2 / float(iteration))

def test(epoch, num_views, iter_vis=50, prefix=''):
	val_loss = 0
	avg_lossl1 = 0
	avg_lossl2 = 0
	avg_loss3 = 0
	num_yeys = 0
	num = 0
	val_accuracy = np.zeros((5,))
	# if model.mean is None:
	# 	# didn't save properly : ugh -- assume was created from the first epoch
	# 	training_data_loader, _ = get_dataset(0, dataset_id=0)
	# 	print('Had to redo ... BUG', model.mean)
	if opt.test_angles:
		_, testing_data_loader = get_dataset(epoch, dataset_id=5)
	else:
		_, testing_data_loader = get_dataset(epoch, dataset_id=3)


	results = torch.zeros((len(testing_data_loader)*opt.testBatchSize,2+1+opt.num_views)).cuda()
	dx = 0
	dy = 0
	u_num = 0
	total_num = 0
	last_iteration = - 100 * 100
	for iteration, batch in enumerate(testing_data_loader, 1):
		
		(inputs, trans_inputs), input_segs, target, pred_depths, gt_depths, thetas, feature_vector, target_theta = get_batch_gt(batch, model, num_views=num_views, requires_grad=False, volatile=True)
		
		pred_depths = list(pred_depths)

		gt_depths = list(gt_depths)

		if opt.loss == 'DepthSilLoss':
			pred_depths[0] = pred_depths[0] * (input_segs[0] > 0.5).float()
			gt_depths[0] = gt_depths[0] * (input_segs[0] > 0.5).float()

			i = 0

			if MEAN_DEPTH:
				mean_pred_depth = pred_depths[0].sum(dim=3).sum(dim=2).sum(dim=1) / (input_segs[0] > 0.5).float().sum(dim=3).sum(dim=2).sum(dim=1).clamp(min=1)
				mean_gt_depth = gt_depths[0].sum(dim=3).sum(dim=2).sum(dim=1) / (input_segs[0] > 0.5).float().sum(dim=3).sum(dim=2).sum(dim=1).clamp(min=1)
				
				pred_depths[0] = pred_depths[0] - Variable(mean_pred_depth.unsqueeze(1).unsqueeze(1).unsqueeze(1).data)
				gt_depths[0] = gt_depths[0] - Variable(mean_gt_depth.unsqueeze(1).unsqueeze(1).unsqueeze(1).data)

				pred_depths[0] = pred_depths[0] * (input_segs[0] > 0.5).float()
				gt_depths[0] = gt_depths[0] * (input_segs[0] > 0.5).float()


			loss1 = (nn.L1Loss(reduce=False).cuda()(pred_depths[i] , gt_depths[i]).sum(dim=3).sum(dim=2).sum(dim=1) /(input_segs[i] > 0.5).float().sum(dim=3).sum(dim=2).sum(dim=1).clamp(min=1))
			temp_path = "/scratch/local/ssd/ow/oc_pytorch/src/image_based/%s_%s%s_width%s_trimap_%s" % (opt.loss, str(opt.lr), opt.copy_weights, str(WIDTH_OUTPUT), str(TRIMAP_SIZE)) + "/test_epoch%d__iter%d_loss%s" % (epoch, iteration, opt.loss)
			
			if opt.test_angles:
				print(loss1.size(), results[(iteration-1)*opt.testBatchSize:(iteration-1)*opt.testBatchSize+loss1.size(0),0].size())
				results[(iteration-1)*opt.testBatchSize:(iteration-1)*opt.testBatchSize+loss1.size(0),0] = loss1

			loss1 = loss1.mean()
			loss2, iou, predicted_silhouette, iou_by_id = silhouette_prediction(target_theta, feature_vector, target)

			loss = loss1 * opt.lambda1 + loss2 * opt.lambda2
			

			if opt.test_angles:
				results[(iteration-1)*opt.testBatchSize:(iteration-1)*opt.testBatchSize+iou_by_id.size(0),1] = iou_by_id.squeeze()

				
				for theta_i, theta in enumerate(thetas):
					theta = theta.squeeze()
					results[(iteration-1)*opt.testBatchSize:(iteration-1)*opt.testBatchSize+iou_by_id.size(0), 2+theta_i] = torch.atan2(theta[:,0], theta[:,1])*180/np.pi

				target_theta = target_theta.squeeze()
				results[(iteration-1)*opt.testBatchSize:(iteration-1)*opt.testBatchSize+iou_by_id.size(0), -1] = torch.atan2(target_theta[:,0], target_theta[:,1])*180/np.pi 

			avg_lossl1 += loss1.cpu().data[0] * predicted_silhouette.size(0)
			avg_lossl2 += iou * predicted_silhouette.size(0)
			total_num = total_num + predicted_silhouette.size(0)
		
		optimizer.zero_grad()

		if opt.test_angles:
			np.save('results_nv%d.npy' % opt.max_theta, results.cpu().data.numpy())
		num = num + input_segs[0].size(0)
		print("===> ({}/{}): L1: {:.4f}; Sil IoU: {:.4f}".format(
			iteration, len(testing_data_loader), avg_lossl1 / float(total_num), avg_lossl2 / float(total_num)))
	

def checkpoint(epoch):
	if not(opt.use_large == 'hourglass'):
		dict = {'epoch' : epoch, 'state_dict' : model.state_dict(), 'silhouette_state' : silhouette_prediction.state_dict(), 'stats' : model.val_stats, 'train_stats' : model.stats, 'optimizer' : optimizer.state_dict(), 'mean' : model.mean, 'angle_encoder' : angle_encoder.state_dict()}
	else:
		dict = {'epoch' : epoch, 'state_dict' : model.state_dict(), 'stats' : model.val_stats, 'train_stats' : model.stats, 'optimizer' : optimizer.state_dict()}
		
	model_out_path = "{}model_epoch_{}.pth".format(opt.model_epoch_path, epoch)
	print(os.path.exists(opt.model_epoch_path))
	print(model.val_stats)
	print(model.stats)
	if not(os.path.exists(opt.model_epoch_path)):
		os.makedirs(opt.model_epoch_path)
	torch.save(dict, model_out_path)
	print("Checkpoint saved to {}".format(model_out_path))

	# Check if new best one:
	if len(model.val_stats['loss']) > 0 and (model.val_stats['loss'].argmin() == ((model.val_stats['loss'].size) - 1)):
		shutil.copyfile(model_out_path, "{}model_epoch_{}.pth".format(opt.model_epoch_path, 'best'))
	
	# remove all previous ones with a worse validation loss
	for i in range(0, epoch-1):	
		#if model.val_stats['loss'][i] >=  model.val_stats['loss'][epoch-1] and \
		if os.path.exists("{}model_epoch_{}.pth".format(opt.model_epoch_path, i)):
			os.remove( "{}model_epoch_{}.pth".format(opt.model_epoch_path, i))

import torch.optim.lr_scheduler as lr_scheduler
if opt.use_large == 'hourglass':
	scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10)
else:
	scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10)

last_epoch = -1
best_epoch = 1000
if not(opt.test):
	percentage = 1
	for epoch in range(opt.continue_epoch,500):
		
		if opt.cuda:
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
			silhouette_prediction.load_state_dict(checkpoint_file['silhouette_state'])
			if POST_DEPTH:
				angle_encoder.load_state_dict(checkpoint_file['angle_encoder'])
			if not (opt.use_large == 'hourglass'):
				model.mean = checkpoint_file['mean']
			model.val_stats = checkpoint_file['stats']
			model.stats = checkpoint_file['train_stats']

			optimizer.load_state_dict(checkpoint_file['optimizer'])
			print("Loaded checkpoint")

		model = model.cuda()
		silhouette_prediction = silhouette_prediction.cuda()
		angle_encoder = angle_encoder.cuda()

		model.train()
		
		tloss, tl1, tl2 = train_2views(epoch, opt.num_views, percentage, num_to_use=-1)
		with torch.no_grad():
			model.eval()
		
			vloss, vl1, vl2 = val(45, opt.num_views, iter_vis=50, prefix=opt.suffix)
		scheduler.step(vloss)
		writer.add_scalars('TrainVal/loss', {'train' : tloss, 'val' : vloss}, epoch)
		writer.add_scalars('TrainVal/l1', {'train' : tl1, 'val' : vl1}, epoch)
		writer.add_scalars('TrainVal/l2', {'train' : tl2, 'val' : vl2}, epoch)
		
		checkpoint(epoch)
else:
	model.eval()
	checkpoint_file = torch.load(opt.old_model)
	if 'mean' in checkpoint_file.keys() and not(opt.use_large == 'hourglass'):
		model.mean = checkpoint_file['mean']
		model.load_state_dict(checkpoint_file['state_dict'])
		model.val_stats = checkpoint_file['stats']
		silhouette_prediction.load_state_dict(checkpoint_file['silhouette_state'])
	elif not(opt.copy_weights):
		model.load_state_dict(checkpoint_file['state_dict'])
		model.val_stats = checkpoint_file['stats']
	else:
		model.load_state_dict(checkpoint_file)

	print(">>>>>>> Loading model")
	print("Loaded checkpoint")
	model.loss = opt.loss
	model = model.cuda()
	silhouette_prediction = silhouette_prediction.cuda()
	angle_encoder = angle_encoder.cuda()
	silhouette_prediction.compute_stats = True

	model.eval()
	silhouette_prediction.eval()
	angle_encoder.eval()
	print(model.val_stats)
	print(model.val_stats['loss'].shape)

	with torch.no_grad():
		test(opt.epoch, opt.num_views, iter_vis=1, prefix='analysis' + opt.dataset)
