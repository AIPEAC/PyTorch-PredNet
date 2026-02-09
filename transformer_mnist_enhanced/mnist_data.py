import hickle as hkl
import numpy as np
import torch
import torch.utils.data as data



class MNIST(data.Dataset):
	#TODO: Moving MNIST - dataset for video sequences of moving digits
	def __init__(self, datafile, nt, output_mode='error', N_seq=None, shuffle=False, data_format='channels_first'):
		"""
		Arguments
		---------
		datafile: str
			path to data file containing Moving MNIST sequences
		nt: int
			number of frames in a sequence (should match data nt)
		output_mode: str
			['error', 'prediction']
		N_seq: int
			size of the subset of sequences to use
		shuffle: boolean
			toggle shuffle before taking subset
		data_format: str
			['channels_first', 'channels_last']
		"""

		self.datafile = datafile
		#TODO: Moving MNIST - load sequences directly, shape: (num_sequences, nt, H, W, C)
		self.X = hkl.load(self.datafile)
		self.nt = nt
		self.data_format = data_format
		assert output_mode in {'error', 'prediction'}, 'output_mode must be in {error, prediction}'			
		self.output_mode = output_mode

		#TODO: Moving MNIST - sequences are already in (num_seq, nt, H, W, C) format
		if self.data_format == 'channels_first':
			# Convert from (num_seq, nt, H, W, C) to (num_seq, nt, C, H, W)
			self.X = np.transpose(self.X, (0, 1, 4, 2, 3))
		
		self.im_shape = self.X[0, 0].shape  # Get shape of first frame of first sequence

		#TODO: Moving MNIST - use all sequences as possible starts
		possible_starts = np.arange(self.X.shape[0])

		if shuffle:
			possible_starts = np.random.permutation(possible_starts)
		if N_seq is not None and len(possible_starts) > N_seq:
			possible_starts = possible_starts[:N_seq]
		self.possible_starts = possible_starts
		self.N_sequences = len(self.possible_starts)

	def preprocess(self, X):
		return X.astype(np.float32) / 255.
    
	def __getitem__(self, index):
		#TODO: Moving MNIST - get entire sequence at this index
		loc = self.possible_starts[index]
		# X shape: (nt, C, H, W) for channels_first
		X = self.preprocess(self.X[loc])
		if self.output_mode == 'error':
			y = np.zeros((1,), np.float32)
		elif self.output_mode == 'prediction':
			y = X
		return (X, y)

	def __len__(self):
		return self.N_sequences
