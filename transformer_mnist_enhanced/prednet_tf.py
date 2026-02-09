import torch
import torch.nn as nn
from torch.nn import functional as F
from conv_lstm_cell_x import ConvLSTMCell
from transformer_block_tf import TransformerBlock
from torch.autograd import Variable

class PredNet(nn.Module):
	def __init__(self, input_size, R_channels, A_channels, p_max=1.0, output_mode='error',
				gating_mode='mul', extrap_start_time=None, peephole=True, lstm_tied_bias=True,
				use_transformer=True, num_transformer_heads=4):
		"""
		Arguments
		---------

		input_size: (int, int)
			dimensions of frames (required for peephole connections)
		R_channels:
			number of channel for RNN in each layer
		A_channels:
			number of channels in A_l in each layer
		p_max: float
			Maximum pixel value in input images
		output_mode: str
			Controls what is outputted by Prednet.
			Either 'error', 'prediction', 'all', or layer specification.
			If 'error' the mean response of the error (E) units of each layer will be outputted.
				That is, the output shape will be (batch_size, n_layers, nt).
			If 'prediction', the frame prediction will be outputted.
			If 'pred+err' the output will be the frame predicition concatenated with the mean layer-errors over time
				The frame prediction is flattened before concatenation
			If '<unit_type>+<layer_num>, the features of a particular layer will be outputted
				Unit types are 'R', 'Ahat', 'A', and 'E'
				Ex: 'Ahat2' is the prediction generated at the third layer
		gating_mode: str
			Controls the gating operation for the ConvLSTM cells
			Either 'mul' or 'sub' (multiplicative vs. subtractive)
		extrap_start_time: int
			Frame to begin using past predictions as ground-truth
		peephole: boolean
			To include/exclude peephole connections in ConvLSTM
		lstm_tied_bias: boolean
			To use tied/untied bias in ConvLSTM convolutions
		use_transformer: boolean
			#TODO: PredNet Transformer - Enable/disable Transformer fusion
		num_transformer_heads: int
			#TODO: PredNet Transformer - Number of attention heads for each transformer layer
		"""
		
		super(PredNet, self).__init__()
		self.r_channels = R_channels + (0, )  # for convenience
		self.a_channels = A_channels
		self.n_layers = len(R_channels)
		self.input_size = input_size
		self.output_mode = output_mode
		self.gating_mode = gating_mode
		self.extrap_start_time = extrap_start_time
		self.peephole = peephole
		self.lstm_tied_bias = lstm_tied_bias
		self.p_max = p_max
		self.use_transformer = use_transformer
		self.num_transformer_heads = num_transformer_heads
		
		# Input validity checks
		default_output_modes = ['prediction', 'error', 'pred+err']
		layer_output_modes = [unit + str(l) for l in range(self.n_layers) for unit in ['R', 'E', 'A', 'Ahat']]
		default_gating_modes = ['mul', 'sub']
		assert output_mode in default_output_modes + layer_output_modes, 'Invalid output_mode: ' + str(output_mode)
		assert gating_mode in default_gating_modes, 'Invalid gating_mode: ' + str(gating_mode)
		
		if self.output_mode in layer_output_modes:
			self.output_layer_type = self.output_mode[:-1]
			self.output_layer_num = int(self.output_mode[-1])
		else:
			self.output_layer_type = None
			self.output_layer_num = None

		h, w = self.input_size

		for i in range(self.n_layers):
			# A_channels multiplied by 2 because E_l concactenates pred-target and target-pred
			# Hidden states don't have same size due to upsampling
			# How does this handle i = L-1 (final layer) | appends a zero

			if self.gating_mode == 'mul':	
				cell = ConvLSTMCell((h, w), 2 * self.a_channels[i] + self.r_channels[i+1], self.r_channels[i],
									(3, 3), gating_mode='mul', peephole=self.peephole, tied_bias=self.lstm_tied_bias)
			elif self.gating_mode == 'sub':
				cell = ConvLSTMCell((h, w), 2 * self.a_channels[i] + self.r_channels[i+1], self.r_channels[i],
									(3, 3), gating_mode='sub', peephole=self.peephole, tied_bias=self.lstm_tied_bias)

			setattr(self, 'cell{}'.format(i), cell)
			h = h // 2
			w = w // 2

		for i in range(self.n_layers):
			# Calculate predictions A_hat
			conv = nn.Sequential(nn.Conv2d(self.r_channels[i], self.a_channels[i], 3, padding=1), nn.ReLU())
			setattr(self, 'conv{}'.format(i), conv)

		self.upsample = nn.Upsample(scale_factor=2)
		self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

		for l in range(self.n_layers - 1):
			# Propagate error as next layer's target (line 16 of Lotter algo)
			# In channels = 2 * A_channels[l] because of pos/neg error concat
			# NOTE: Operation belongs to curr layer l and produces next layer  state l+1

			update_A = nn.Sequential(nn.Conv2d(2* self.a_channels[l], self.a_channels[l+1], (3, 3), padding=1), self.maxpool)
			setattr(self, 'update_A{}'.format(l), update_A)

		# #TODO: PredNet Transformer - Initialize independent transformer blocks for each layer
		if self.use_transformer:
			# Pre-calculate spatial dimensions for each layer
			h, w = self.input_size
			spatial_dims = []
			for i in range(self.n_layers):
				spatial_dims.append((h, w))
				h = h // 2
				w = w // 2

			# #TODO: PredNet Transformer - Create independent transformers for E, R, Ahat at each layer
			for l in range(self.n_layers):
				# E layer: 2 * a_channels[l] channels
				e_dim = 2 * self.a_channels[l]
				e_heads = self._get_valid_num_heads(e_dim, num_transformer_heads)
				e_transformer = TransformerBlock(e_dim, num_heads=e_heads)
				setattr(self, 'transformer_E{}'.format(l), e_transformer)

				# R layer: r_channels[l] channels
				r_dim = self.r_channels[l]
				r_heads = self._get_valid_num_heads(r_dim, num_transformer_heads)
				r_transformer = TransformerBlock(r_dim, num_heads=r_heads)
				setattr(self, 'transformer_R{}'.format(l), r_transformer)

			# #TODO: PredNet Transformer - Learnable fusion weights alpha for E and R only (removed Ahat)
			for l in range(self.n_layers):
				# Independent alpha for E, R at each layer (Ahat removed as it's not used)
				setattr(self, 'alpha_E{}'.format(l), nn.Parameter(torch.tensor(0.45)))
				setattr(self, 'alpha_R{}'.format(l), nn.Parameter(torch.tensor(0.45)))
	
	def _get_valid_num_heads(self, input_dim, num_heads):
		"""
		Helper function to find the largest num_heads that divides input_dim.
		
		Arguments:
		-----------
		input_dim: int
			Input dimension for transformer
		num_heads: int
			Desired number of heads
		
		Returns:
		--------
		valid_num_heads: int
			The largest num_heads <= desired num_heads that divides input_dim
		"""
		# #TODO: PredNet Transformer - Find valid num_heads that divides input_dim
		if input_dim % num_heads == 0:
			return num_heads
		# Find largest divisor of input_dim that is <= num_heads
		for heads in range(min(num_heads, input_dim), 0, -1):
			if input_dim % heads == 0:
				return heads
		return 1

	def set_output_mode(self, output_mode):
		"""
		set_output_mode:
			Resets output mode
		Arguments
		_________
		output_mode: str
		"""
		
		# Change output mode
		self.output_mode = output_mode

		# Input validity checks
		default_output_modes = ['prediction', 'error', 'pred+err']
		layer_output_modes = [unit + str(l) for l in range(self.n_layers) for unit in ['R', 'E', 'A', 'Ahat']]
		assert output_mode in default_output_modes + layer_output_modes, 'Invalid output_mode: ' + str(output_mode)
		
		if self.output_mode in layer_output_modes:
			self.output_layer_type = self.output_mode[:-1]
			self.output_layer_num = int(self.output_mode[-1])
		else:
			self.output_layer_type = None
			self.output_layer_num = None
	
	def _apply_transformer_to_feature_map(self, feature_map, transformer_block):
		"""
		Helper function to apply transformer to a convolutional feature map.
		
		Arguments:
		-----------
		feature_map: Tensor
			Shape (batch_size, channels, height, width)
		transformer_block: TransformerBlock
			The transformer block to apply
		
		Returns:
		--------
		output: Tensor
			Transformed feature map of same shape as input
		"""
		batch_size, channels, height, width = feature_map.shape
		
		# #TODO: PredNet Transformer - Reshape feature map to (batch, height*width, channels) for transformer
		# Treat each spatial position as a token in the sequence
		feature_flat = feature_map.permute(0, 2, 3, 1)  # (batch, height, width, channels)
		feature_flat = feature_flat.reshape(batch_size, height*width, channels)  # (batch, seq_len, channels)
		
		# #TODO: PredNet Transformer - Apply transformer block
		transformed = transformer_block(feature_flat)
		
		# #TODO: PredNet Transformer - Reshape back to (batch, channels, height, width)
		transformed = transformed.reshape(batch_size, height, width, channels)
		transformed = transformed.permute(0, 3, 1, 2)  # (batch, channels, height, width)
		
		return transformed
	
	def step(self, a, states):
		"""
		step:
		Performs inference for a single time step

		Arguments:
		_________
		a: Tensor
			target image frame
		states: list
			contains layer states -->  [R + C + E]
			if self.extrap_start_time --> [R + C + E, prev_pred, t]
		"""
		
		batch_size = a.size(0)
		device = a.device  # Get device from input tensor
		R_layers = states[:self.n_layers]
		C_layers = states[self.n_layers:2*self.n_layers]
		E_layers = states[2*self.n_layers:3*self.n_layers]
		
		#TODO: PredNet Transformer - List to store raw errors for Loss calculation (preventing trivial solution)
		E_loss_layers = [None] * self.n_layers

		if self.extrap_start_time is not None:
			t = states[-1]
			if t >= self.extrap_start_time: # if past self.extra_start_time use previous prediction as input
				a = states[-2]

		# Update representation units
		for l in reversed(range(self.n_layers)):
			cell = getattr(self, 'cell{}'.format(l))
			r_tm1 = R_layers[l]
			c_tm1 = C_layers[l]
			e_tm1 = E_layers[l]
			if l == self.n_layers - 1:
				r, c = cell(e_tm1, (r_tm1, c_tm1))
			else:
				tmp = torch.cat((e_tm1, self.upsample(R_layers[l+1])), 1)
				r, c = cell(tmp, (r_tm1, c_tm1))
			R_layers[l] = r
			C_layers[l] = c

		# Perform error forward pass
		for l in range(self.n_layers):
			conv = getattr(self, 'conv{}'.format(l))
			a_hat = conv(R_layers[l])
			if l == 0:
				a_hat= torch.min(a_hat, torch.tensor(self.p_max, device=device)) # alternative SatLU (Lotter)
				frame_prediction = a_hat
			pos = F.relu(a_hat - a)
			neg = F.relu(a - a_hat)
			e = torch.cat([pos, neg],1)
			
			#TODO: PredNet Transformer - Store raw error for Loss calculation (Ground Truth for optimization)
			# Even if 'e' is modified by Transformer for state updates, we must minimize the REAL error.
			E_loss_layers[l] = e
			
			# #TODO: PredNet Transformer - Apply transformer and fusion for E and R only (Ahat removed)
			if self.use_transformer:
				transformer_E = getattr(self, 'transformer_E{}'.format(l))
				transformer_R = getattr(self, 'transformer_R{}'.format(l))

				# Apply transformers to E and R components
				e_transformed = self._apply_transformer_to_feature_map(e, transformer_E)
				r_transformed = self._apply_transformer_to_feature_map(R_layers[l], transformer_R)

				# #TODO: PredNet Transformer - Fuse transformer outputs with original outputs using independent learnable weights for E and R
				alpha_e = torch.sigmoid(getattr(self, 'alpha_E{}'.format(l)))
				alpha_r = torch.sigmoid(getattr(self, 'alpha_R{}'.format(l)))
				
				e = alpha_e * e_transformed + (1 - alpha_e) * e
				R_layers[l] = alpha_r * r_transformed + (1 - alpha_r) * R_layers[l]
			
			E_layers[l] = e
			
			# Handling layer-specific outputs
			if self.output_layer_num == l:
				if self.output_layer_type == 'A':
					output = a
				elif self.output_layer_type == 'Ahat':
					output = a_hat
				elif self.output_layer_type == 'R':
					output = R_layers[l]
				elif self.output_layer_type == 'E':
					#TODO: PredNet Transformer - Return raw error for specific layer monitoring
					output = E_loss_layers[l]

			if l < self.n_layers - 1: # updating A for next layer
				update_A = getattr(self, 'update_A{}'.format(l))
				a = update_A(e)

		if self.output_layer_type is None:
			if self.output_mode == 'prediction':
				output = frame_prediction
			else:
				# Batch flatten (return 2D matrix) then mean over units
				# Finally, concatenate layers (batch, n_layers)
				#TODO: PredNet Transformer - Use reshape instead of view for non-contiguous tensors
				#TODO: PredNet Transformer - Use E_loss_layers (Raw Error) for Loss Output
				# This forces the model to minimize the actual pixel difference, not the transformed(suppressed) one.
				mean_E_layers = torch.cat([torch.mean(e.reshape(batch_size, -1), axis=1, keepdim=True) for e in E_loss_layers], axis=1)
				if self.output_mode == 'error':
					output = mean_E_layers
				else:
					output = torch.cat([frame_prediction.reshape(batch_size, -1), mean_E_layers], axis=1)

		states = R_layers + C_layers + E_layers
		if self.extrap_start_time is not None:
			states += [frame_prediction, t+1]
		return output, states

	def forward(self, input):
		"""
		forward:

		Perform inference on a sequence of frames

		Arguments:
		input: Tensor
			A (batch_size, nt, num_channels, height, width) tensor
		"""

		R_layers = [None] * self.n_layers
		C_layers = [None] * self.n_layers
		E_layers = [None] * self.n_layers

		h, w = self.input_size # input.size(-2), input.size(-1)
		batch_size = input.size(0)
		device = input.device  # Get device from input tensor

		# Initialize states on the same device as input
		for l in range(self.n_layers):
			R_layers[l] = torch.zeros(batch_size, self.r_channels[l], h, w, requires_grad=True, device=device)
			C_layers[l] = torch.zeros(batch_size, self.r_channels[l], h, w, requires_grad=True, device=device)
			E_layers[l] = torch.zeros(batch_size, 2*self.a_channels[l], h, w, requires_grad=True, device=device)
			# Size of hidden state halves from each layer to the next
			h = h//2
			w = w//2

		states = R_layers + C_layers + E_layers
		# Initialize previous_prediction
		if self.extrap_start_time is not None:
			frame_prediction = torch.zeros_like(input[:,0], dtype=torch.float32)
			states += [frame_prediction, -1] # [a, t]
			
		num_time_steps = input.size(1)
		total_output = [] # contains output sequence
		for t in range(num_time_steps):
			a = input[:,t].type(torch.FloatTensor).to(device)
			output, states = self.step(a, states)
			total_output.append(output)

		ax = len(output.shape)
		# print(output.shape)
		#TODO: PredNet Transformer - Use reshape instead of view for non-contiguous tensors
		total_output = [out.reshape(out.shape + (1,)) for out in total_output]
		total_output = torch.cat(total_output, axis=ax) # (batch, ..., nt)
		return total_output

