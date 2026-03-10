# PredNet PyTorch Implementation

## References
- Paper: [Deep Predictive Coding Networks for Video Prediction and Unsupervised Learning](https://arxiv.org/abs/1605.08104)
- Original Keras code: [coxlab/prednet](https://github.com/coxlab/prednet)
- Original PyTorch code: [leido/pytorch-prednet](https://github.com/leido/pytorch-prednet)

---

## Implementation
### Directory
- [prednet_pytorch_mnist/](prednet_pytorch_mnist/)

#### Training
- File: [mnist_train_all.py](prednet_pytorch_mnist/mnist_train_all.py)
- Parameters:
  - num_epochs: 6
  - batch_size: 2
  - lr: 0.001
  - nt: 20 (sequence length)
  - n_train_seq: 7000
  - n_val_seq: 1000
#### Dataset
- File: [mnist_data.py](prednet_pytorch_mnist/mnist_data.py)
- Parameters:
  - nt: 20 (frames per sequence)
  - image_size: (64, 64)
  - channels: 3 (RGB)
#### Model Hyperparameters
- loss_mode: 'L_0' or 'L_all' (default: 'L_all')
- peephole: False
- lstm_tied_bias: False
- gating_mode: 'mul' or 'sub' (default: 'mul')
- A_channels & R_channels: (3, 48, 96, 192)

#### Output
- Best model: [prednet-L_all-mul-peepFalse-tbiasFalse-best.pt](prednet_pytorch_mnist/models/prednet-L_all-mul-peepFalse-tbiasFalse-best.pt)
- Loss history: [original_mnist-prednet-L_all-mul-peepFalse-tbiasFalse-loss_history.jsonl](data_compare/loss_history/original_mnist-prednet-L_all-mul-peepFalse-tbiasFalse-loss_history.jsonl)


