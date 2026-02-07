# Prednet-Transformer
PredNet, implemented with pytorch.

## Setup
### download a ffmpeg

- Download [ffmpeg](https://ffmpeg.org/download.html). Unzip the file to a folder and add ffmpeg to system PATH
	- [Windows download](https://www.gyan.dev/ffmpeg/builds/)

### download Moving
- Download [MNIST](https://www.cs.toronto.edu/~nitish/unsupervised_video/mnist_test_seq.npy). Put it in Mnist_npy_data

## Important Data
- Total data: Standard Moving MNIST, 10000 sequences (7000 train, 1000 validation, 2000 test) × 20 frames per sequence
- Data ratios: Train:Validation:Test = 7:1:2
- Total epoch: 150
    - learning rate: 0.001 (first 75 epochs) → 0.0001 (final 75 epochs)
    - Each epoch: all 7000 train sequences + all 1000 validation sequences
- Testing:
    - Frames 0-9 (first 10 frames): provide actual data to model (teacher forcing)
    - Frames 10-19 (last 10 frames): model generates predictions without actual data (autoregressive, self-supervised)
    - Evaluation: separate evaluation code computes MSE/error metrics between predicted frames 10-19 and ground truth


