# PredNet PyTorch Implementation

## References
- Paper: [Deep Predictive Coding Networks for Video Prediction and Unsupervised Learning](https://arxiv.org/abs/1605.08104)
- Original Keras code: [coxlab/prednet](https://github.com/coxlab/prednet)
- Original PyTorch code: [leido/pytorch-prednet](https://github.com/leido/pytorch-prednet)

---

## Implementation

  [Directory](/predenet_pytorch_mnist/)

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


