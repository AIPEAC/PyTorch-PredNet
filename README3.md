# PredNet-Transformer Sparse Version
PredNet with sparse Transformer integration, implemented in PyTorch.

## Setup
### Download FFmpeg

- Download [ffmpeg](https://ffmpeg.org/download.html). Unzip the file to a folder and add ffmpeg to system PATH
	- [Windows download](https://www.gyan.dev/ffmpeg/builds/)

### Download Moving MNIST

- Download [MNIST](https://www.cs.toronto.edu/~nitish/unsupervised_video/mnist_test_seq.npy). Put it in `/mnist_npy_data`. Run [process_mnist.py](mnist_npy_data/process_mnist.py) to create hickle files for subsequent testing.

## Training Configuration
- **Total epochs**: 1 (single epoch for testing)
- **Batch size**: 2 (reduced from 8 for GPU memory efficiency)
- **Learning rate**: 0.001
- **Timesteps per sequence**: 20 frames
- **Data**: Standard Moving MNIST, 10000 sequences (7000 train, 1000 validation, 2000 test)

## Transformer Configuration (Sparse)
- **Transformer Component**: Only Layer 0 E (error) enhanced with Transformer
- **Application Frequency**: Every 3 timesteps (t=0, 3, 6, 9, ...) → ~7 transformer operations per 20-step sequence
- **Memory Optimization**:
  - FeedForward dimension: 2× input_dim (vs 4× standard)
  - Layer 0 E spatial size: 64×64 (4096 tokens)
- **Architecture**: 
  - Single TransformerBlock per Layer 0 E
  - 6 attention heads (divisor of 6-channel E at Layer 0)

## Sparse Transformer Integration

### Design Rationale
- **Why Layer 0 E only**: Reduces transformer computation from 12 blocks (full) to 1 block (sparse)
- **Why every 3 timesteps**: Further reduces from 20 to ~7 transformer applications per sequence
- **Why 2× ff_dim**: Halves FeedForward parameter count (3,145,728 → 1,572,864 params in Layer 0 E transformer)

### Fusion Mechanism
At timesteps where transformer is applied (t % 3 == 0):

$$E^{\text{fused}}_{\text{Layer0},t} = \sigma(\alpha_E) \cdot \text{Transformer}(E_{\text{Layer0},t}) + (1-\sigma(\alpha_E)) \cdot E_{\text{Layer0},t}$$

Where:
- σ = sigmoid activation
- α_E = learnable fusion weight per layer
- Allows model to learn optimal balance between original and transformer-enhanced error

### Feedback Path
Fused E is used:
1. **Next layer calculation**: A_{Layer1,t} = update_A(E_fused_{Layer0,t})
2. **Next timestep**: As E_prev in ConvLSTM for Layer 1 at t+1

### Output
- **Model file**: `prednet-tf-sparse-L_all-mul-peepFalse-tbiasFalse.pt`
- **Best model**: `prednet-tf-sparse-L_all-mul-peepFalse-tbiasFalse-best.pt`
- **Loss history**: `data_compare/loss_history/transformer_mnist-prednet-tf-sparse-...-loss_history.json`
- **Fusion weight**: `data_compare/loss_history/transformer_mnist-prednet-tf-sparse-...-alpha_E0.json`

## Performance Notes
- **Memory footprint**: ~25% of full transformer version due to sparse application and reduced batch size
- **Computational cost**: ~40 transformer operations per epoch (vs 240 with full transformer)
- **Training observation**: Single epoch to verify convergence and fusion weight learning (alpha_E0)
