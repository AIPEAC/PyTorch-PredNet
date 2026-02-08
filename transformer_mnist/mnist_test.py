from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import json

import os
import hickle as hkl

import torch
from torch.utils.data import DataLoader
import numpy as np

#TODO: MNIST - import MNIST instead of KITTI
from mnist_data import MNIST
#TODO: MNIST - import from mnist_settings
from mnist_settings import *
from prednet_tf_sparse import PredNet 

# Visualization parameters
n_plot = 4 # number of plot to make (must be <= batch_size)
#TODO: MNIST - import from mnist_settings for DATA_DIR
from mnist_settings import DATA_DIR

# Model parameters
gating_mode = 'mul'
peephole = False
lstm_tied_bias = False
nt = 20  # #TODO: Moving MNIST - changed to match training (was 10)
extrap_start_time = 10  #TODO: Moving MNIST - frames 0-9 use real data, frames 10-19 are autoregressive predictions
batch_size = 2

#TODO: MNIST - changed from (3, 48, 96, 192) to match single grayscale channel input
default_channels = (3, 48, 96, 192)
#TODO: MNIST - updated channel configuration for 3-channel RGB
channel_six_layers = (3, 48, 96, 192, 384, 768)
#TODO: MNIST - changed A_channels from 1 to 3 for RGB images
A_channels = (3, 48, 96, 192)
#TODO: MNIST - changed R_channels from 1 to 3 for RGB images
R_channels = (3, 48, 96, 192)
using_default_channels = A_channels == default_channels
num_layers = len(A_channels)

test_file = os.path.join(DATA_DIR, 'X_test.hkl')
#TODO: MNIST - removed test_sources, no source tracking needed

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
model_name = 'prednet-tf-sparse-L_all-mul-peepFalse-tbiasFalse-best'  # #TODO: Change the model to other models
model_file = os.path.join(MODEL_DIR, model_name + '.pt')

RESULTS_SAVE_DIR = './'

#TODO: MNIST - MNIST constructor takes no test_sources parameter
mnist_test = MNIST(test_file, nt, output_mode='prediction')
num_steps = len(mnist_test)//batch_size
test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=False)

#TODO: Moving MNIST - image size is 64x64 instead of 128x160
input_size = mnist_test.im_shape[1:3] #(64, 64)

model = PredNet(input_size, R_channels, A_channels, output_mode='prediction', gating_mode=gating_mode,
				extrap_start_time=extrap_start_time, peephole=peephole, lstm_tied_bias=lstm_tied_bias,
				use_transformer=True, num_transformer_heads=4)
model.load_state_dict(torch.load(model_file))

print('Model: ' + model_name)
if torch.cuda.is_available():
	print('Using GPU.')
	model.cuda()

pred_MSE = 0.0
copy_last_MSE = 0.0
sequence_mses = []  # #TODO: Moving MNIST - record per-sequence MSE
seq_idx = 0
for step, (inputs, targets) in enumerate(test_loader):
	# ---------------------------- Test Loop -----------------------------
	# print(f"inputs {inputs.shape}")
	inputs = inputs.cuda() # batch x time_steps x channel x width x height
	
	targets = targets

	pred = model(inputs) # (batch_size, channels, width, height, nt)
	pred = pred.cpu()
	pred = pred.permute(0,4,1,2,3) # (batch_size, nt, channels, width, height)

	if step == 0:	
		print('inputs: ', inputs.size())
		print('targets: ', targets.size())
		print('predicted: ', pred.size(), pred.dtype)

	#TODO: Moving MNIST - calculate per-sequence MSE
	for i in range(targets.shape[0]):
		seq_mse = torch.mean((targets[i, 1:] - pred[i, 1:])**2).item()
		sequence_mses.append({'sequence_idx': seq_idx, 'mse': seq_mse})
		seq_idx += 1

	pred_MSE += torch.mean((targets[:, 1:] - pred[:, 1:])**2).item() # look at all timesteps after the first
	copy_last_MSE += torch.mean((targets[:, 1:] - targets[:, :-1])**2).item()
	
	#TODO: MNIST - test full dataset - removed step == 20 condition and break to test all sequences
	if step == num_steps - 1: # Plot predictions from last batch only
		# Plot some predictions
		targets = targets.detach().numpy() * 255.
		pred = pred.detach().numpy() * 255.

		targets = np.transpose(targets, (0, 1, 3, 4, 2))
		pred = np.transpose(pred, (0, 1, 3, 4, 2))
		
		targets = targets.astype(int)
		pred = pred.astype(int)

		aspect_ratio = float(pred.shape[2]) / pred.shape[3]
		fig = plt.figure(figsize = (nt, 2*aspect_ratio))
		gs = gridspec.GridSpec(2, nt)
		gs.update(wspace=0., hspace=0.)
		plot_save_dir = os.path.join(RESULTS_SAVE_DIR, 'prediction_plots/') # NOTE: images saved to transformer_mnist/prediction_plots/
		if not os.path.exists(plot_save_dir): os.mkdir(plot_save_dir)
		plot_idx = np.random.permutation(targets.shape[0])[:n_plot]
		for i in plot_idx:
			for t in range(nt):
				plt.subplot(gs[t])
				#TODO: MNIST - using grayscale colormap for single channel images
				plt.imshow(targets[i,t], interpolation='none', cmap='gray') # requires 'channel_last'
				plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
				if t==0: plt.ylabel('Actual', fontsize=10)

				plt.subplot(gs[t + nt])
				#TODO: MNIST - using grayscale colormap for single channel images
				plt.imshow(pred[i,t], interpolation='none', cmap='gray')
				plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
				if t==0: plt.ylabel('Predicted', fontsize=10)
			
			img_filename = model_name + '-step_' + str(step)
			if extrap_start_time is not None:
				img_filename = img_filename + '-extrap_' + str(extrap_start_time) + '+' + str(nt - extrap_start_time)
			img_filename = img_filename + '-plot_' + str(i)

			while os.path.exists(img_filename + '.png'):
				img_filename += '_'
			
			print('Saving ' + img_filename)
			img_filename = plot_save_dir + img_filename
			plt.savefig(img_filename + '.png')
			plt.clf()
			print(f'Image Saved as {img_filename}.png')

# Calculate dataset MSE
pred_MSE /= num_steps
copy_last_MSE /= num_steps

print('Prediction MSE: {:.6f}'.format(pred_MSE)) # no need to worry about "first time step"
print('Copy-Last MSE: {:.6f}'.format(copy_last_MSE))

#TODO: Moving MNIST - save per-sequence MSE to json in data_compare/test_plots
data_compare_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data_compare', 'test_plots')
os.makedirs(data_compare_dir, exist_ok=True)
test_mse_file = 'transformer_mnist-' + model_name + '-test_sequence_mse.json'
test_mse_path = os.path.join(data_compare_dir, test_mse_file)
with open(test_mse_path, 'w') as f:
	json.dump(sequence_mses, f, indent=2)
print(f'Sequence MSE saved to {test_mse_path}')
