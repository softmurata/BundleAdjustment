# The method for running a regression on the inner layers of the network
import sys
sys.path.append('./oc_pytorch/src/')
sys.path.append('./oc_pytorch/src/image_based/')
sys.path.append('./oc_pytorch/src/utils/')

from utils.Globals import *

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import argparse

import sklearn
from sklearn.linear_model import Lasso, LinearRegression, RidgeCV

from net_layers.SkipNet import Pix2PixModel
from MeshBasedPrediction import SilhouettePrediction, AngleEncoder
from data_utils.data import *


parser = argparse.ArgumentParser(description='PyTorch encoder/3D decoder')
parser.add_argument('--old_model', type=str, default='/scratch/shared/slow/ow/python_epochs/sculpture_ijcv_gtdepth_augdata_synthsculp_minmax/sculpture/MeshBasedPrediction_sgd/pix2pix_none/num_views2_batchsize_16finetuneFalse_loss_DepthSilLoss_lambda1_21.0_1.0DepthSilLoss_0.001_size_pix2pix_weightminmaxmodel_epoch_best.pth')
parser.add_argument('--imdb', type=str, default='/scratch/local/ssd/ow/')
parser.add_argument('--threads', type=int, default=10, help='number of threads for data loader to use')
parser.add_argument('--testBatchSize', type=int, default=16, help='testing batch size')
parser.add_argument('--num_views', type=int, default=2, help='testing batch size')

opt = parser.parse_args() 


index = 1
use_input = True
use_fv = False
use_fv_up_1 = False
use_fv_up_2 = False
use_angle_encoder_1 = False
use_angle_encoder_2 = False

if use_angle_encoder_1:
	folder = 'dtheta_%d_avg_angleencoder1' % index
elif use_angle_encoder_2:
	folder = 'dtheta_%d_avg_angleencoder2' % index
elif use_fv_up_2:
	folder = 'dtheta_%d_avg_fv_up_2' % index
elif use_fv_up_1:
	folder = 'dtheta_%d_avg_fv_up_1' % index
elif use_fv:
	folder = 'dtheta_%d_avg_fv' % index
elif use_input:
	folder = 'dtheta_%d_avg_input' % index
else:
	folder = 'dtheta_%d_avg' % index

if not os.path.exists('/scratch/local/ssd/ow/results_ijcv_experiments/%s/' % folder):
	os.makedirs('/scratch/local/ssd/ow/results_ijcv_experiments/%s/' % folder)

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

	if POST_DEPTH:
		target_theta = angle_encoder(target_theta)

	heights = model(*input_args)
	fv_silhouette = heights[-1]
	heights = heights[:-1]

	return (orig_images, inputs + [target_img]), masks, target, heights, gt_prediction, thetas, fv_silhouette, target_theta

def get_batch_gt_controltheta(batch, model, theta, num_views=2, volatile=False, requires_grad=False, use_identity=False):
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

	from data_utils.load_functions import get_sin_cos

	input_args[-2] = Variable(torch.Tensor(get_sin_cos(theta))).cuda().unsqueeze(0).unsqueeze(3)
	input_args[-1] = angle_encoder(thetas[-2].unsqueeze(3))

	if POST_DEPTH:
		target_theta = angle_encoder(target_theta)

	heights = model(*input_args)
	fv_silhouette = heights[-1]
	heights = heights[:-1]

	return (orig_images, inputs + [target_img]), masks, target, heights, gt_prediction, thetas, fv_silhouette, target_theta

def determine_p_values(fvs, theta_prime, fv_vals, theta_prime_val):
	from linear_models import LinearRegressionStatsModels

	model = LinearRegressionStatsModels()
	p_values = model.fit(fvs, theta_prime[:,0])

	if not os.path.exists('/scratch/local/ssd/ow/results_ijcv_experiments/%s_vp' % folder):
		os.makedirs('/scratch/local/ssd/ow/results_ijcv_experiments/%s_vp'  % folder)
	np.savetxt('/scratch/local/ssd/ow/results_ijcv_experiments/%s_vp/p_0_values.csv' % folder, p_values, delimiter=',')
	p_values = model.fit(fvs, theta_prime[:,1])

	if not os.path.exists('/scratch/local/ssd/ow/results_ijcv_experiments/%s_vp'  % folder):
		os.makedirs('/scratch/local/ssd/ow/results_ijcv_experiments/%s_vp' % folder)
	np.savetxt('/scratch/local/ssd/ow/results_ijcv_experiments/%s_vp/p_1_values.csv' % folder, p_values, delimiter=',')
	

def determine_num_vp_independent(fvs, theta_prime, fv_vals, theta_prime_val):
	alphas = 15

	all_acc_abs_traina = np.zeros((alphas,2))
	all_acc_pow_traina = np.zeros((alphas,2))
	all_acc_absa = np.zeros((alphas,2))
	all_acc_powa = np.zeros((alphas,2))
	num_zero = np.zeros((alphas,2))

	iter_num = 0
	while iter_num < alphas:
		# First get the result using standard linear regression
		regressor = Lasso(fit_intercept=True, normalize=True, alpha=1e-5, tol=1e-10)


		if fvs.shape[1] <= 1:
			print(np.zeros((fvs.shape[0], 2)).shape, theta_prime.shape)
			regressor.fit(np.zeros((fvs.shape[0], 2)), theta_prime)
			theta_pred_sincos = regressor.predict(np.zeros((fv_vals.shape[0], 2)))
			theta_pred = np.arctan2(theta_pred_sincos[:,0], theta_pred_sincos[:,1])
			theta_pred_train_sincos = regressor.predict(np.zeros((fvs.shape[0], 2)))
			theta_pred_train = np.arctan2(theta_pred_train_sincos[:,0], theta_pred_train_sincos[:,1])
		else:
			print(fvs.shape)
			regressor.fit(fvs, theta_prime)
			theta_pred_sincos = regressor.predict(fv_vals)
			theta_pred = np.arctan2(theta_pred_sincos[:,0], theta_pred_sincos[:,1])
			theta_pred_train_sincos = regressor.predict(fvs)
			theta_pred_train = np.arctan2(theta_pred_train_sincos[:,0], theta_pred_train_sincos[:,1])


		theta_primegt = np.arctan2(theta_prime[:,0], theta_prime[:,1])
		theta_prime_valgt = np.arctan2(theta_prime_val[:,0], theta_prime_val[:,1])

		all_acc_abs_train = (theta_pred_train - theta_primegt)
		all_acc_abs_train += (all_acc_abs_train > np.pi).astype(np.int32) * - 2 * np.pi + (all_acc_abs_train < -np.pi).astype(np.int32) * 2 * np.pi
		all_acc_pow_train = np.sqrt((all_acc_abs_train ** 2).mean())
		all_acc_abs_train = all_acc_abs_train.mean()
		
		all_acc_abs = (theta_pred - theta_prime_valgt)
		all_acc_abs += (all_acc_abs > np.pi).astype(np.int32) * -2 * np.pi + (all_acc_abs < -np.pi).astype(np.int32) * 2 * np.pi
		all_acc_pow= np.sqrt((all_acc_abs ** 2).mean())
		all_acc_abs = all_acc_abs.mean()

		all_acc_abs_traina[iter_num,:] = all_acc_abs_train
		all_acc_pow_traina[iter_num,:] = all_acc_pow_train
		all_acc_absa[iter_num,:] = all_acc_abs
		all_acc_powa[iter_num,:] = all_acc_pow
		
		if fvs.shape[1] <= 1:
			num_zero[iter_num,:] = num_zero[iter_num-1,:]
			iter_num += 1
			continue

		if iter_num == 0:
			num_zero[iter_num,:] = np.sum((regressor.coef_[0,:] > 1e-10) | (regressor.coef_[1,:] > 1e-10))
		else:
			num_zero[iter_num,:] = num_zero[iter_num-1,:] + np.sum((regressor.coef_[0,:] > 1e-10) | (regressor.coef_[1,:] > 1e-10))

		# Remove the unnecessary ones
		print(regressor.coef_.shape, fvs.shape)
		print(np.sum(regressor.coef_ > 1e-10, axis=1))
		print(all_acc_powa)
		fvs = fvs[:,(regressor.coef_[0,:] < 1e-10) & (regressor.coef_[1,:] < 1e-10)] 
		fv_vals = fv_vals[:,(regressor.coef_[0,:] < 1e-10) & (regressor.coef_[1,:] < 1e-10)] 
		iter_num += 1

	np.savetxt('/scratch/local/ssd/ow/results_ijcv_experiments/%s_vp/all_acc_abs.csv' % folder, all_acc_absa, delimiter=',')
	np.savetxt('/scratch/local/ssd/ow/results_ijcv_experiments/%s_vp/all_acc_pow.csv' % folder, all_acc_powa, delimiter=',')
	np.savetxt('/scratch/local/ssd/ow/results_ijcv_experiments/%s_vp/all_acc_abs_train.csv' % folder, all_acc_abs_traina, delimiter=',')
	np.savetxt('/scratch/local/ssd/ow/results_ijcv_experiments/%s_vp/all_acc_pow_train.csv' % folder, all_acc_pow_traina, delimiter=',')
	np.savetxt('/scratch/local/ssd/ow/results_ijcv_experiments/%s_vp/num_zero.csv' % folder, num_zero, delimiter=',')
	

def run_regressor_on_fvs(fvs, theta_prime, fv_vals, theta_prime_val):

	# Create a graph based on how well the given value can be predicted as a function of the alpha value
	# And the corresponding number of non-zero components of the model as well as the accuracy of the
	# prediction on both train / test
	alphas = [] #[1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1]

	all_acc_abs_train = np.zeros((len(alphas)+1,1))
	all_acc_pow_train = np.zeros((len(alphas)+1,1))
	all_acc_abs = np.zeros((len(alphas)+1,1))
	all_acc_pow = np.zeros((len(alphas)+1,1))
	num_zero = np.zeros((len(alphas)+1,2))
	alpha = np.zeros((len(alphas)+1,))

	# First get the result using standard linear regression
	regressor = RidgeCV(fit_intercept=True)
	regressor.fit(fvs, theta_prime)
	theta_pred_sincos = regressor.predict(fv_vals)
	print(theta_pred_sincos.shape, theta_prime_val.shape)
	theta_pred = np.arctan2(theta_pred_sincos[:,0], theta_pred_sincos[:,1])
	theta_pred_train_sincos = regressor.predict(fvs)
	theta_pred_train = np.arctan2(theta_pred_train_sincos[:,0], theta_pred_train_sincos[:,1])

	print(theta_prime.shape, theta_prime_val.shape)
	theta_prime = np.arctan2(theta_prime[:,0], theta_prime[:,1])
	theta_prime_val = np.arctan2(theta_prime_val[:,0], theta_prime_val[:,1])

	print(theta_prime.shape, theta_prime_val.shape)

	print((theta_prime_val - theta_prime.mean()).mean())
	print(((theta_prime_val - theta_prime.mean()) ** 2).mean())

	all_acc_abs_train = (theta_pred_train - theta_prime)
	all_acc_abs_train += (all_acc_abs_train > np.pi).astype(np.int32) * - 2 * np.pi + (all_acc_abs_train < -np.pi).astype(np.int32) * 2 * np.pi
	all_acc_pow_train = np.sqrt((all_acc_abs_train ** 2).mean())
	all_acc_abs_train = all_acc_abs_train.mean()

	all_acc_abs = (theta_pred - theta_prime_val)
	all_acc_abs += (all_acc_abs > np.pi).astype(np.int32) * -2 * np.pi + (all_acc_abs < -np.pi).astype(np.int32) * 2 * np.pi
	all_acc_pow= np.sqrt((all_acc_abs ** 2).mean())
	all_acc_abs = all_acc_abs.mean()
	
	num_zero[0,:] = np.sum(regressor.coef_ < 1e-10, axis=1)

	temp_coefficients = regressor.coef_
	temp_best = all_acc_abs

	# for alpha_ind in range(0, len(alphas)):
	# 	# Then modify the alpha value
	# 	regressor = Lasso(fit_intercept=True, normalize=True, alpha=alphas[alpha_ind], tol=1e-10)
	# 	regressor.fit(fvs, theta_prime)

	# 	# Return accuracy and coefficients
	# 	theta_pred = regressor.predict(fv_vals)
	# 	theta_pred_train = regressor.predict(fvs)

	# 	pred = np.abs(theta_pred - theta_prime_val).mean(axis=0).sum()
	# 	if pred < temp_best:
	# 		temp_coefficients = regressor.coef_

	# 	all_acc_abs_train[alpha_ind+1,:] = np.abs(theta_pred_train - theta_prime).mean(axis=0)
	# 	all_acc_pow_train[alpha_ind+1,:] = ((theta_pred_train - theta_prime) ** 2).mean(axis=0)


	# 	acc_abs = np.abs(theta_pred - theta_prime_val).mean(axis=0)
	# 	acc_pow = ((theta_pred - theta_prime_val) ** 2).mean(axis=0)

	# 	all_acc_abs[alpha_ind+1,:] = acc_abs
	# 	all_acc_pow[alpha_ind+1,:] = acc_pow
	# 	num_zero[alpha_ind+1,:] = np.sum(regressor.coef_ < 1e-10, axis=1)
	# 	alpha[alpha_ind+1] = alphas[alpha_ind]

	print(alpha)
	print(all_acc_abs)
	print(all_acc_pow)
	print(all_acc_abs_train)
	print(all_acc_pow_train)
	print(num_zero)

	np.savetxt('/scratch/local/ssd/ow/results_ijcv_experiments/%s/ridgecv_alpha.csv' % folder, alpha, delimiter=',')
	np.savetxt('/scratch/local/ssd/ow/results_ijcv_experiments/%s/ridgecv_all_acc_abs.csv' % folder, np.array([all_acc_abs]), delimiter=',')
	np.savetxt('/scratch/local/ssd/ow/results_ijcv_experiments/%s/ridgecv_all_acc_pow.csv' % folder, np.array([all_acc_pow]), delimiter=',')
	np.savetxt('/scratch/local/ssd/ow/results_ijcv_experiments/%s/ridgecv_all_acc_abs_train.csv' % folder, np.array([all_acc_abs_train]), delimiter=',')
	np.savetxt('/scratch/local/ssd/ow/results_ijcv_experiments/%s/ridgecv_all_acc_pow_train.csv' % folder, np.array([all_acc_pow_train]), delimiter=',')
	np.savetxt('/scratch/local/ssd/ow/results_ijcv_experiments/%s/ridgecv_num_zero.csv' % folder, num_zero, delimiter=',')
	np.savetxt('/scratch/local/ssd/ow/results_ijcv_experiments/%s/ridgecv_coefficients' % folder, temp_coefficients)

	print(folder, index)

def get_theta_to_use(target_theta, input_thetas):
	if index >= 0:
		return input_thetas[index].squeeze().data.cpu().numpy()

	else:
		return target_theta.squeeze().data.cpu().numpy()

def visualise_latent_embedding(model, silhouette_model):
	test_set = get_sculpture_set([3], opt.imdb, False, random=45, num_views=opt.num_views, scale=256,  mean=model.mean)
	test_set.views = range(0,opt.num_views)
	testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)

	# Stores the output state from the variables
	class OutputState:
		pass

	temp_output = OutputState()

	SIZE = 544

	def get_result(self, input, output):
		print(input[0].size())
		temp_output.temp = input[0]

	silhouette_model.decoder.model[0].register_forward_hook(get_result)

	img = np.zeros((544,36))

	for iteration_to_use in [1, 24, 41, 101]:
		for theta in range(0, 36):
			for iteration, batch in enumerate(testing_data_loader, 1):

				if not iteration == iteration_to_use:
					continue
				theta_rad = np.pi * theta * 10. / 180.
				(inputs, trans_inputs), input_segs, target, pred_depths, gt_depths, thetas, feature_vector, target_theta = get_batch_gt_controltheta(batch, model, theta_rad, num_views=opt.num_views, requires_grad=False, volatile=True)

				loss2, predicted_silhouette = silhouette_model(target_theta, feature_vector, target)
				
				img[:,theta] = temp_output.temp.contiguous().squeeze().data.cpu().numpy()
				break
		

		np.save('/scratch/local/ssd/ow/results_ijcv_experiments/max_fv%d' % iteration_to_use, img)
		print(img)

def analyse_network(model, silhouette_model, desired_layer):
	# Analyses the network
	train_set = get_sculpture_set([1], opt.imdb, False, random=45, num_views=opt.num_views, scale=256,  mean=model.mean)
	training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)

	test_set = get_sculpture_set([3], opt.imdb, False, random=45, num_views=opt.num_views, scale=256,  mean=model.mean)
	test_set.views = range(0,opt.num_views)
	testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)


	# Stores the output state from the variables
	class OutputState:
		pass

	temp_output = OutputState()

	MULTIPLES = 1
	if use_angle_encoder_1:
		SIZE = 32
		fvs = np.zeros((len(train_set)*MULTIPLES, SIZE))
		fv_vals = np.zeros((len(test_set), SIZE))
	elif use_angle_encoder_2:
		SIZE = 32
		fvs = np.zeros((len(train_set)*MULTIPLES, SIZE))
		fv_vals = np.zeros((len(test_set), SIZE))


	elif use_fv_up_2:
		SIZE = 512*16
		fvs = np.zeros((len(train_set)*MULTIPLES, SIZE))
		fv_vals = np.zeros((len(test_set), SIZE))

	elif use_fv_up_1:
		SIZE = 512*4
		fvs = np.zeros((len(train_set)*MULTIPLES, SIZE))
		fv_vals = np.zeros((len(test_set), SIZE))

	elif use_fv:
		SIZE = 512
		fvs = np.zeros((len(train_set)*MULTIPLES, 512))
		fv_vals = np.zeros((len(test_set), 512))
	elif use_input:
		SIZE = 512
		fvs = np.zeros((len(train_set)*MULTIPLES, 512))
		fv_vals = np.zeros((len(test_set), 512))
	else:
		SIZE = 1024
		fvs = np.zeros((len(train_set)*MULTIPLES, 1024))
		fv_vals = np.zeros((len(test_set), 1024))
	theta_prime = np.zeros((len(train_set)*MULTIPLES, 2))
	theta_prime_val = np.zeros((len(test_set), 2))

	def get_result(self, input, output):
		if use_input:
			temp_output.temp = input[0].squeeze()[:,0:512].contiguous()
		else:
			print(output.size())
			temp_output.temp = output

	#model.netG.model.submodule.submodule.submodule.submodule.down[1].register_forward_hook(get_result)
	#print(angle_encoder)
	#print(model.netG.model.submodule.submodule.submodule.submodule.submodule.angle_fv[0])

	if use_angle_encoder_1:
		model.netG.model.submodule.submodule.submodule.submodule.submodule.angle_fv[0].register_forward_hook(get_result)
	elif use_angle_encoder_2:
		model.netG.model.submodule.submodule.submodule.submodule.submodule.angle_fv[2].register_forward_hook(get_result)
	elif use_fv_up_2:
		model.netG.model.submodule.submodule.submodule.submodule.submodule.down[1].register_forward_hook(get_result)
	elif use_fv_up_1:
		model.netG.model.submodule.submodule.submodule.submodule.submodule.submodule.down[1].register_forward_hook(get_result)
	elif use_fv:
		model.netG.model.submodule.submodule.submodule.submodule.submodule.submodule.submodule.down[1].register_forward_hook(get_result)
	else:
		silhouette_model.decoder.model[0].register_forward_hook(get_result)
	#print('1 asdf 1')
	
	# Get the train set
	offset = 0
	for i in range(0, MULTIPLES):
		train_set.views = np.random.choice(range(0,4), opt.num_views+1, replace=False)
		for iteration, batch in enumerate(training_data_loader, 1):
			print(iteration, len(training_data_loader))
			(inputs, trans_inputs), input_segs, target, pred_depths, gt_depths, thetas, feature_vector, target_theta = get_batch_gt(batch, model, num_views=opt.num_views, requires_grad=False, volatile=True)
			loss2, predicted_silhouette = silhouette_model(target_theta, feature_vector, target)
			
			fvs[offset:offset+temp_output.temp.size(0),:] = temp_output.temp.view(temp_output.temp.size(0), SIZE).contiguous().data.cpu().numpy()
			theta_prime[offset:offset + temp_output.temp.size(0)] = get_theta_to_use(target_theta, thetas) 

			offset = offset + temp_output.temp.size(0)
			
			
			
	# Get the test set
	offset = 0
	for iteration, batch in enumerate(testing_data_loader, 1):
		print(iteration, len(testing_data_loader))
		(inputs, trans_inputs), input_segs, target, pred_depths, gt_depths, thetas, feature_vector, target_theta = get_batch_gt(batch, model, num_views=opt.num_views, requires_grad=False, volatile=True)
		loss2, predicted_silhouette = silhouette_model(target_theta, feature_vector, target)
		
		fv_vals[offset:offset+temp_output.temp.size(0),:] = temp_output.temp.view(temp_output.temp.size(0), SIZE).contiguous().data.cpu().numpy()
		theta_prime_val[offset:offset + temp_output.temp.size(0)] = get_theta_to_use(target_theta, thetas) 

		offset = offset + temp_output.temp.size(0)
		
		

	#determine_p_values(fvs, theta_prime, fv_vals, theta_prime_val)
	run_regressor_on_fvs(fvs, theta_prime, fv_vals, theta_prime_val)

checkpoint = torch.load(opt.old_model)


angle_encoder = AngleEncoder().cuda()
silhouette_model = SilhouettePrediction().cuda()
model = Pix2PixModel('', 0).cuda()
model.mean = checkpoint['mean']

angle_encoder.load_state_dict(checkpoint['angle_encoder'])
silhouette_model.load_state_dict(checkpoint['silhouette_state'])
model.load_state_dict(checkpoint['state_dict'])

model.eval()
silhouette_model.eval()

#analyse_network(model, silhouette_model, '', )
visualise_latent_embedding(model, silhouette_model)
