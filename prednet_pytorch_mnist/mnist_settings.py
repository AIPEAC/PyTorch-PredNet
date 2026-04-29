#TODO: MNIST uses 28x28 grayscale images, pointing to preprocessed MNIST data directory
# Where MNIST data will be saved if you run process_mnist.py
# If you directly download the processed data, change to the path of the data.
import os
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'mnist_npy_data')
