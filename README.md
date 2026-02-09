# PredNet PyTorch Implementation

## References
- Paper: [Deep Predictive Coding Networks for Video Prediction and Unsupervised Learning](https://arxiv.org/abs/1605.08104)
- Original Keras code: [coxlab/prednet](https://github.com/coxlab/prednet)
- Original PyTorch code: [leido/pytorch-prednet](https://github.com/leido/pytorch-prednet)

---

## Original Implementation (KITTI Dataset)

### Directory: original_mnist/

#### Training
- File: [mnist_train_all.py](original_mnist/mnist_train_all.py)
- Parameters:
  - num_epochs: 150
  - batch_size: 4
  - lr: 0.001
  - nt: 10 (sequence length)
  - n_train_seq: 500
  - n_val_seq: 100

#### Model Hyperparameters
- loss_mode: 'L_0' or 'L_all' (default: 'L_0')
- peephole: False
- lstm_tied_bias: False
- gating_mode: 'mul' or 'sub' (default: 'mul')
- A_channels & R_channels: (3, 48, 96, 192)

#### Output
- Best model: [models/prednet-L_all-mul-peepFalse-tbiasFalse-best.pt](original_mnist/models/prednet-L_all-mul-peepFalse-tbiasFalse-best.pt)
- Loss history: [history/prednet-L_all-mul-peepFalse-tbiasFalse-param_history.jsonl](original_mnist/history/prednet-L_all-mul-peepFalse-tbiasFalse-param_history.jsonl)

---

## Transformer Sparse Version (Moving MNIST)

### Directory: 	transformer_mnist/

#### Setup Requirements
1. FFmpeg: [Download](https://www.gyan.dev/ffmpeg/builds/) and add to system PATH
2. Moving MNIST: [Download dataset](https://www.cs.toronto.edu/~nitish/unsupervised_video/mnist_test_seq.npy), place in /mnist_npy_data, run [process_mnist.py](mnist_npy_data/process_mnist.py)

#### Training Configuration
- File: [mnist_train_all_sparse.py](transformer_mnist/mnist_train_all_sparse.py)
- Epochs: 1
- Batch size: 2
- Learning rate: 0.001
- Timesteps per sequence: 20 frames
- Dataset: Moving MNIST (7000 train, 1000 validation, 2000 test)

#### Transformer Configuration (Sparse)
- Component: Layer 0 E (error) with Transformer enhancement
- Application frequency: Every 3 timesteps (t=0, 3, 6, 9, ...)
- Transformer operations per sequence: ~7 per 20-step sequence
- FeedForward dimension: 2× input_dim
- Architecture: Single TransformerBlock per Layer 0 E
- Attention heads: 6
- Layer 0 E spatial size: 64×64 (4096 tokens)

#### Fusion Mechanism
E_Layer0_t_fused = sigmoid(alpha_E) * Transformer(E_Layer0_t) + (1 - sigmoid(alpha_E)) * E_Layer0_t

- alpha_E: learnable fusion weight per layer (initialized: 0.45)
- Applied when t % 3 == 0

#### Output
- Model file: [models/prednet-tf-sparse-L_all-mul-peepFalse-tbiasFalse.pt](transformer_mnist/models/)
- Loss history: [data_compare/loss_history/](data_compare/loss_history/)


