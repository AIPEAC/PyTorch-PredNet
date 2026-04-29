from __future__ import print_function
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from mnist_data import MNIST
from mnist_settings import *
from prednet_x import PredNet

def init_weights(m):
	""""
	init_weights:
	Initialize sub-module weights using xavier_uniform (for kernels) and 0. (for kernel biases)

	Arguments:
	_________
	m: nn.Module
	"""

	if isinstance(m, nn.Conv2d):
		# Reset kernels and biases
		nn.init.xavier_uniform_(m.weight)
		if m.bias is not None:
			nn.init.zeros_(m.bias)

#TODO: Original MNIST - Function to extract and record all model parameters
def record_parameters_to_file(model, epoch, step, batch_idx, param_file):
	"""Extract all model parameters and append to json file (efficient, no memory accumulation)"""
	param_record = {
		'epoch': epoch,
		'step': step,
		'batch_idx': batch_idx,
		'parameters': {}
	}
	
	# Record all parameters
	for name, param in model.named_parameters():
		if param.requires_grad:
			values = param.data.cpu().detach().numpy()
			# Store statistics instead of full tensor to save memory
			param_record['parameters'][name] = {
				'mean': float(values.mean()),
				'std': float(values.std()),
				'min': float(values.min()),
				'max': float(values.max()),
				'norm': float(values.reshape(-1).dot(values.reshape(-1)))**0.5,
				'shape': list(values.shape)
			}
	
	# Append to file (line by line, not accumulating in memory)
	with open(param_file, 'a') as f:
		f.write(json.dumps(param_record) + '\n')

#TODO: Original MNIST Training - Function to save batch prediction visualization every 100 batches
def save_batch_prediction(vis_model, inputs, epoch, step, history_dir, model_name, nt):
	"""Save prediction visualization for first sequence in batch every 100 steps"""
	with torch.no_grad():
		vis_model.eval()
		# Check if inputs and model are on same device
		device = next(vis_model.parameters()).device
		if inputs.device != device:
			inputs = inputs.to(device)
			
		pred = vis_model(inputs)  # (batch_size, channels, width, height, nt)
		vis_model.train()  # Switch back to train mode
	
	# inputs is (batch, nt, channels, width, height)
	targets = inputs  # batch x nt x channels x width x height
	pred = pred.cpu()
	pred = pred.permute(0, 4, 1, 2, 3)  # (batch_size, nt, channels, width, height)
	
	# Only visualize first sequence in batch
	targets = targets[0:1].cpu()  # Keep batch dimension: (1, nt, channels, width, height)
	pred = pred[0:1]  # (1, nt, channels, width, height)
	
	# Convert to numpy and scale
	targets = targets.detach().numpy() * 255.
	pred = pred.detach().numpy() * 255.
	
	# Transpose to (batch, nt, width, height, channels)
	targets = np.transpose(targets, (0, 1, 3, 4, 2))
	pred = np.transpose(pred, (0, 1, 3, 4, 2))
	
	targets = targets.astype(int)
	pred = pred.astype(int)
	
	i = 0  # Only one sequence (first one)
	aspect_ratio = float(pred.shape[2]) / pred.shape[3]
	fig = plt.figure(figsize=(nt, 2*aspect_ratio))
	gs = gridspec.GridSpec(2, nt)
	gs.update(wspace=0., hspace=0.)
	
	for t in range(nt):
		plt.subplot(gs[t])
		# Extract first channel for 3-channel RGB images
		plt.imshow(targets[i,t,:,:,0], interpolation='none', cmap='gray')
		plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
		if t==0: plt.ylabel('Input', fontsize=10)
		
		plt.subplot(gs[t + nt])
		# Extract first channel for 3-channel RGB images
		plt.imshow(pred[i,t,:,:,0], interpolation='none', cmap='gray')
		plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
		if t==0: plt.ylabel('Predicted', fontsize=10)
	
	img_filename = f'{model_name}-epoch{epoch:02d}-step{step:05d}'
	pred_plot_dir = os.path.join(history_dir, 'prediction_plots')
	os.makedirs(pred_plot_dir, exist_ok=True)
	img_path = os.path.join(pred_plot_dir, img_filename)
	
	plt.savefig(img_path + '.png')
	plt.clf()
	plt.close(fig) # Close figure to free memory
	print(f'Prediction saved: {img_path}.png')

# Training parameters
num_epochs = 150
# TODO: MNIST - set batch size to 16 from 4
batch_size = 16 # 16
lr = 0.001 # if epoch < 75 else 0.0001
nt = 20 # num of time steps #TODO: Moving MNIST - changed from 10 to 20 frames
n_train_seq = 7000 #TODO: Moving MNIST - use entire training set (7000 sequences) per epoch
n_val_seq = 1000 #TODO: Moving MNIST - use entire validation set (1000 sequences) per epoch

# Model parameters
loss_mode = 'L_0'
peephole = False
lstm_tied_bias = False
gating_mode = 'mul'

#TODO: MNIST - set to (3, 48, 96, 192) for 3-channel RGB Moving MNIST input
default_channels = (3, 48, 96, 192)
A_channels = (3, 48, 96, 192)
R_channels = (3, 48, 96, 192)
using_default_channels = A_channels == default_channels
num_layers = len(A_channels)

#TODO: MNIST - device detection for CPU/GPU compatibility
device = 'cuda'

if loss_mode == 'L_0':
	layer_loss_weights = torch.zeros((num_layers, 1), device='cuda')
	layer_loss_weights[0,0] = 1.
elif loss_mode == 'L_all':
	layer_loss_weights = 0.1 * torch.ones((num_layers, 1), device='cuda')
	layer_loss_weights[0] = 1.

time_loss_weights = 1./(nt - 1) * torch.ones((nt, 1), device='cuda') # lambda_t's in Lotter et al. 2017
time_loss_weights[0] = 0

# Directories

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

train_file = os.path.join(DATA_DIR, 'X_train.hkl')
#TODO: MNIST - removed train_sources, no source tracking needed
val_file = os.path.join(DATA_DIR, 'X_val.hkl')
#TODO: MNIST - removed val_sources, no source tracking needed

mnist_train = MNIST(train_file, nt, output_mode='error', N_seq=n_train_seq)
mnist_val = MNIST(val_file, nt, output_mode='error',  N_seq=n_val_seq)
#TODO: Moving MNIST - image size is 64x64 instead of 128x160
input_size = mnist_train.im_shape[1:3] #(64, 64)
num_train_steps = len(mnist_train)//batch_size

train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(mnist_val, batch_size=batch_size, shuffle=True)

model = PredNet(input_size, R_channels, A_channels, output_mode='error', gating_mode=gating_mode,
				peephole=peephole, lstm_tied_bias=lstm_tied_bias)

print('Using GPU.')
model.cuda()

#TODO: Original MNIST Training - Create visualization model for prediction during training
vis_model = PredNet(input_size, R_channels, A_channels, output_mode='prediction', gating_mode=gating_mode,
				peephole=peephole, lstm_tied_bias=lstm_tied_bias)

if torch.cuda.is_available():
	vis_model.cuda()

model.apply(init_weights)
vis_model.apply(init_weights)

if using_default_channels:
	model_name = 'prednet-{}-{}-peep{}-tbias{}'.format(loss_mode, gating_mode, peephole, lstm_tied_bias)
else:
	channels_str = '_'.join([str(x) for x in A_channels])
	model_name = 'prednet-{}-{}-peep{}-tbias{}-chans_{}'.format(loss_mode, gating_mode, peephole, lstm_tied_bias, channels_str)

print('Model: ' + model_name)

optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-07)
criterion = nn.L1Loss()

def lr_scheduler(optimizer, epoch):
	if epoch < num_epochs //2:
		return optimizer
	else:
		for param_group in optimizer.param_groups:
			param_group['lr'] = 0.0001
		return optimizer

min_val_loss = float('inf')

#TODO: Training Loss - Setup loss tracking file in jsonl format (append mode to reduce memory)
loss_history_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data_compare', 'loss_history')
os.makedirs(loss_history_dir, exist_ok=True)
if using_default_channels:
	loss_history_file = 'original_mnist-' + 'prednet-{}-{}-peep{}-tbias{}'.format(loss_mode, gating_mode, peephole, lstm_tied_bias) + '-loss_history.jsonl'
else:
	channels_str = '_'.join([str(x) for x in A_channels])
	loss_history_file = 'original_mnist-' + 'prednet-{}-{}-peep{}-tbias{}-chans_{}'.format(loss_mode, gating_mode, peephole, lstm_tied_bias, channels_str) + '-loss_history.jsonl'
loss_history_path = os.path.join(loss_history_dir, loss_history_file)
if os.path.exists(loss_history_path):
	os.remove(loss_history_path)  # Clear previous run
print(f'Loss history will be saved to: {loss_history_path}')

#TODO: Original MNIST - Setup parameter tracking file in history directory (jsonl format)
history_dir = os.path.join(os.path.dirname(__file__), 'history')
os.makedirs(history_dir, exist_ok=True)

# Determine base model name for parameter file
if using_default_channels:
	model_name_temp = 'prednet-{}-{}-peep{}-tbias{}'.format(loss_mode, gating_mode, peephole, lstm_tied_bias)
else:
	channels_str = '_'.join([str(x) for x in A_channels])
	model_name_temp = 'prednet-{}-{}-peep{}-tbias{}-chans_{}'.format(loss_mode, gating_mode, peephole, lstm_tied_bias, channels_str)

param_file = os.path.join(history_dir, f'{model_name_temp}-param_history.jsonl')
if os.path.exists(param_file):
	os.remove(param_file)  # Clear previous run
print(f'Parameter history will be saved to: {param_file}')

for epoch in range(num_epochs):

	# ----------------- Training Loop ----------------------
	train_loss = 0.0
	optimizer = lr_scheduler(optimizer, epoch)
	model.train()

	for step, (inputs, targets) in enumerate(train_loader):
		# batch x time_steps x channel x width x height
		inputs = inputs.cuda()
		targets = targets.cuda()
		
		# Refer to Eqn (5) in Lotter et al. 2017
		# L_train = Sum_t( lam_t * Sum_l( lam_l/nl * Sum_{n_l}(E^t_l) ) )
		errors = model(inputs) # batch x n_layers x nt
		
		loc_batch = errors.size(0)
		# Weighted sum of error time-components
		# (batch*n_layers x nt)(nt x 1) -->  batch*n_layers x 1
		loss = torch.mm(errors.view(-1, nt), time_loss_weights) 	
		# Weighted sum of error layer-components
		# (batch x n_layer)(n_layer x 1) --> batch x 1 
		loss = torch.mm(loss.view(loc_batch, -1), layer_loss_weights) 	
		# Average batch los
		# train_loss = torch.mean(train_loss, dim=0, keepdim=True)

		# Calculate Mean Absolute Error
		loss = criterion(loss, targets)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		train_loss += loss.item() * batch_size

		if step % 10 == 0:
			print('step: {}/{}, loss: {:.6f}'.format(step, num_train_steps, loss))
		#TODO: Training Loss - Record batch loss immediately after optimizer step
		with open(loss_history_path, 'a') as f:
			f.write(json.dumps({'epoch': epoch+1, 'step': step, 'batch_loss': loss.item()}) + '\n')
		
		#TODO: Original MNIST - Record all parameters after each batch to avoid memory accumulation
		record_parameters_to_file(model, epoch+1, step, step // batch_size, param_file)
		
		#TODO: Original MNIST Training - Save prediction visualization every 100 batches
		if step % 100 == 0:
			# Share weights from training model to visualization model
			vis_model.load_state_dict(model.state_dict())
			save_batch_prediction(vis_model, inputs, epoch+1, step, history_dir, model_name, nt)

	train_loss /= len(mnist_train)
	print('Epoch: {}/{}, loss: {:.6f}'.format(epoch+1, num_epochs, train_loss)) 
    
	# ------------------  Validation Loop  -------------------
	model.eval()
	val_loss = 0.0
	with torch.no_grad():
		for step, (inputs, targets) in enumerate(val_loader):
			# batch x time_steps x channels x width x heigth
			inputs = inputs.cuda()
			targets = targets.cuda()
			errors = model(inputs) # barch x n_layers x nt
			loc_batch = errors.size(0)
			# Weighted sum of error time-components
			loss = torch.mm(errors.view(-1, nt), time_loss_weights)
			# Weighted sum of error layer-components
			loss = torch.mm(loss.view(loc_batch, -1), layer_loss_weights)
			# Calculate Mean Absolute Error
			loss = criterion(loss, targets)
			val_loss += loss.item() * batch_size

	val_loss /= len(mnist_val)
	print('Validation loss: {:.6f}'.format(val_loss))
	#TODO: Training Loss - Record epoch validation loss at end of epoch
	with open(loss_history_path, 'a') as f:
		f.write(json.dumps({'epoch': epoch+1, 'epoch_train_loss': train_loss, 'epoch_val_loss': val_loss}) + '\n')
	
	if val_loss < min_val_loss:
		print('Validation Loss Decreased: {:.6f} --> {:.6f} \t Saving the Model'.format(min_val_loss, val_loss))
		min_val_loss = val_loss
		# Save model
		torch.save(model.state_dict(), os.path.join(MODEL_DIR, model_name + '-best.pt'))
	print()

# Save model
torch.save(model.state_dict(), os.path.join(MODEL_DIR, model_name + '.pt'))

#TODO: Training Loss - Loss history already saved line-by-line during training
print(f'Loss history saved to {loss_history_path}')
print(f'Parameter history jsonl saved to {param_file}')
