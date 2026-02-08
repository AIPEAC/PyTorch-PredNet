from __future__ import print_function
import os
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from mnist_data import MNIST
from mnist_settings import *
#TODO: Transformer MNIST - Import Transformer-enabled PredNet
from prednet_tf import PredNet

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

#TODO: Transformer MNIST - Function to extract and record all model parameters
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

# Training parameters
num_epochs = 40
batch_size = 8
lr = 0.001 # if epoch < 75 else 0.0001
nt = 20 # num of time steps #TODO: Moving MNIST - changed from 10 to 20 frames
n_train_seq = 7000 #TODO: Moving MNIST - use entire training set (7000 sequences) per epoch
n_val_seq = 1000 #TODO: Moving MNIST - use entire validation set (1000 sequences) per epoch

# Model parameters
loss_mode = 'L_all'
peephole = False
lstm_tied_bias = False
gating_mode = 'mul'

#TODO: MNIST - set to (3, 48, 96, 192) for 3-channel RGB Moving MNIST input
default_channels = (3, 48, 96, 192)
A_channels = (3, 48, 96, 192)
R_channels = (3, 48, 96, 192)
using_default_channels = A_channels == default_channels
num_layers = len(A_channels)

if loss_mode == 'L_0':
	layer_loss_weights = torch.zeros((num_layers, 1), device='cuda')
	layer_loss_weights[0,0] = 1.
elif loss_mode == 'L_all':
	layer_loss_weights = 0.1 * torch.ones((num_layers, 1), device='cuda')
	layer_loss_weights[0] = 1.

time_loss_weights = 1./(nt - 1) * torch.ones((nt, 1), device='cuda') # lambda_t's in Lotter et al. 2017
time_loss_weights[0] = 0

# Directories

train_file = os.path.join(DATA_DIR, 'X_train.hkl')
#TODO: MNIST - removed train_sources, no source tracking needed
val_file = os.path.join(DATA_DIR, 'X_val.hkl')
#TODO: MNIST - removed val_sources, no source tracking needed

mnist_train = MNIST(train_file, nt, output_mode='error', N_seq=n_train_seq)
mnist_val = MNIST(val_file, nt, output_mode='error',  N_seq=n_val_seq)
#TODO: Moving MNIST - image size is 64x64 instead of 128x160
input_size = mnist_train.im_shape[1:3] #(64, 64)
num_train_steps = len(mnist_train)//batch_size

train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
val_loader = DataLoader(mnist_val, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)

model = PredNet(input_size, R_channels, A_channels, output_mode='error', gating_mode=gating_mode,
				peephole=peephole, lstm_tied_bias=lstm_tied_bias,
				#TODO: Transformer MNIST - Enable transformer fusion for R and E only (removed Ahat as it's not used downstream)
				use_transformer=True, num_transformer_heads=4)

if torch.cuda.is_available():
	print('Using GPU.')
	model.cuda()
model.apply(init_weights)

if using_default_channels:
	#TODO: Transformer MNIST - Add -tf suffix to distinguish transformer version
	model_name = 'prednet-tf-{}-{}-peep{}-tbias{}'.format(loss_mode, gating_mode, peephole, lstm_tied_bias)
else:
	channels_str = '_'.join([str(x) for x in A_channels])
	model_name = 'prednet-tf-{}-{}-peep{}-tbias{}-chans_{}'.format(loss_mode, gating_mode, peephole, lstm_tied_bias, channels_str)

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
	loss_history_file = 'transformer_mnist-' + 'prednet-tf-{}-{}-peep{}-tbias{}'.format(loss_mode, gating_mode, peephole, lstm_tied_bias) + '-loss_history.jsonl'
else:
	channels_str = '_'.join([str(x) for x in A_channels])
	loss_history_file = 'transformer_mnist-' + 'prednet-tf-{}-{}-peep{}-tbias{}-chans_{}'.format(loss_mode, gating_mode, peephole, lstm_tied_bias, channels_str) + '-loss_history.jsonl'
loss_history_path = os.path.join(loss_history_dir, loss_history_file)
if os.path.exists(loss_history_path):
	os.remove(loss_history_path)  # Clear previous run
print(f'Loss history will be saved to: {loss_history_path}')

#TODO: Transformer MNIST - Setup parameter tracking file in history directory (jsonl format)
history_dir = os.path.join(os.path.dirname(__file__), 'history')
os.makedirs(history_dir, exist_ok=True)

# Determine base model name for parameter file
if using_default_channels:
	model_name_temp = 'prednet-tf-{}-{}-peep{}-tbias{}'.format(loss_mode, gating_mode, peephole, lstm_tied_bias)
else:
	channels_str = '_'.join([str(x) for x in A_channels])
	model_name_temp = 'prednet-tf-{}-{}-peep{}-tbias{}-chans_{}'.format(loss_mode, gating_mode, peephole, lstm_tied_bias, channels_str)

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
		#TODO: Transformer MNIST - Record all parameters after each batch to avoid memory accumulation
		record_parameters_to_file(model, epoch+1, step, step // batch_size, param_file)

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
	#TODO: Training Loss - Append epoch loss to jsonl file directly to reduce memory
	with open(loss_history_path, 'a') as f:
		f.write(json.dumps({'epoch': epoch+1, 'train_loss': train_loss, 'val_loss': val_loss}) + '\n')
	
	if val_loss < min_val_loss:
		print('Validation Loss Decreased: {:.6f} --> {:.6f} \t Saving the Model'.format(min_val_loss, val_loss))
		min_val_loss = val_loss
		# Save model
		torch.save(model.state_dict(), model_name + '-best.pt')
	print()

# Save model
torch.save(model.state_dict(), model_name + '.pt')

#TODO: Training Loss - Loss history already saved line-by-line during training
print(f'Loss history saved to {loss_history_path}')
print(f'Parameter history jsonl saved to {param_file}')
