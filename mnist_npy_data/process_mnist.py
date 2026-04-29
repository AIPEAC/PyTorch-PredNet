'''
Code for processing downloaded Moving MNIST data to hickle format
'''

import os
import numpy as np
import cv2
import hickle as hkl

#TODO: Moving MNIST - save hkl files in current directory (mnist_npy_data)
DATA_DIR = os.path.dirname(os.path.abspath(__file__))

if not os.path.exists(DATA_DIR): 
	os.makedirs(DATA_DIR, exist_ok=True)

#TODO: Moving MNIST - process downloaded npy file to hkl format
def process_downloaded_moving_mnist(npy_file_path):
	"""
	Convert pre-downloaded Moving MNIST npy file to hickle format with 7:1:2 split
	
	Args:
		npy_file_path: path to the downloaded moving_mnist.npy or mnist_test_seq.npy file
	
	Usage:
		python process_mnist.py path/to/mnist_test_seq.npy
	"""
	if not os.path.exists(npy_file_path):
		print(f'Error: File not found: {npy_file_path}')
		return
	
	#TODO: Moving MNIST - load npy file
	print(f'Loading Moving MNIST from {npy_file_path}...')
	data = np.load(npy_file_path)
	print(f'Loaded data shape: {data.shape}')
	
	#TODO: Moving MNIST - handle different data shapes (20, 10000, 64, 64) format
	# Data shape from mnist_test_seq.npy: (time_steps, num_sequences, height, width)
	# Need to convert to (num_sequences, time_steps, height, width)
	if len(data.shape) == 4 and data.shape[2:] == (64, 64):
		# Transpose from (time_steps, num_sequences, height, width) to (num_sequences, time_steps, height, width)
		sequences = np.transpose(data, (1, 0, 2, 3))
		print(f'Transposed data to (num_sequences, time_steps, height, width): {sequences.shape}')
	elif len(data.shape) == 4:
		# Assume already (num_sequences, seq_length, height, width)
		sequences = data
	elif len(data.shape) == 3:
		sequences = np.expand_dims(data, axis=0)
	else:
		print(f'Unexpected data shape: {data.shape}')
		return
	
	#TODO: Moving MNIST - get number of sequences and time steps
	num_sequences = sequences.shape[0]
	seq_length = sequences.shape[1]
	print(f'Number of sequences: {num_sequences}, Sequence length (frames): {seq_length}')
	
	#TODO: Moving MNIST - resize to 64x64 if needed
	if sequences.shape[2:] != (64, 64):
		print(f'Resizing from {sequences.shape[2:]} to 64x64...')
		resized = np.zeros((num_sequences, seq_length, 64, 64), dtype=np.uint8)
		for i in range(num_sequences):
			for j in range(seq_length):
				resized[i, j] = cv2.resize(sequences[i, j], (64, 64))
		sequences = resized
	
	#TODO: Moving MNIST - apply 7:1:2 split (70% train, 10% val, 20% test)
	train_size = int(0.7 * num_sequences)
	val_size = int(0.1 * num_sequences)  # 1000 sequences for 10000 total
	
	train_sequences = sequences[:train_size]
	val_sequences = sequences[train_size:train_size + val_size]
	test_sequences = sequences[train_size + val_size:]
	
	print(f'Split - Train: {len(train_sequences)}, Val: {len(val_sequences)}, Test: {len(test_sequences)}')
	
	#TODO: Moving MNIST - expand and repeat for RGB compatibility
	train_sequences = np.expand_dims(train_sequences, axis=-1)
	val_sequences = np.expand_dims(val_sequences, axis=-1)
	test_sequences = np.expand_dims(test_sequences, axis=-1)
	
	train_sequences = np.repeat(train_sequences, 3, axis=-1)
	val_sequences = np.repeat(val_sequences, 3, axis=-1)
	test_sequences = np.repeat(test_sequences, 3, axis=-1)
	
	#TODO: Moving MNIST - save as hickle files
	print(f'Saving training sequences: {train_sequences.shape}')
	hkl.dump(train_sequences, os.path.join(DATA_DIR, 'X_train.hkl'))
	
	print(f'Saving validation sequences: {val_sequences.shape}')
	hkl.dump(val_sequences, os.path.join(DATA_DIR, 'X_val.hkl'))
	
	print(f'Saving test sequences: {test_sequences.shape}')
	hkl.dump(test_sequences, os.path.join(DATA_DIR, 'X_test.hkl'))
	
	print('Successfully converted to hickle format!')


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        npy_file = sys.argv[1]
    else:
        #TODO: Moving MNIST - automatically check for npy file in 1MNIST_NPY_DATA directory
        npy_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), '1MNIST_NPY_DATA')
        npy_file = os.path.join(npy_dir, 'mnist_test_seq.npy')
        if not os.path.exists(npy_file):
            print('Usage: python process_mnist.py path/to/mnist_test_seq.npy')
            print(f'Or place mnist_test_seq.npy in {npy_dir}')
            sys.exit(1)
    
    process_downloaded_moving_mnist(npy_file)

