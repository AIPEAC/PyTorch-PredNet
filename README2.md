# Prednet-Transformer
PredNet, implemented with pytorch.

## Setup
### download a ffmpeg

- Download [ffmpeg](https://ffmpeg.org/download.html). Unzip the file to a folder and add ffmpeg to system PATH
	- [Windows download](https://www.gyan.dev/ffmpeg/builds/)

### download Moving
- Download [MNIST](https://www.cs.toronto.edu/~nitish/unsupervised_video/mnist_test_seq.npy). Put it in /mnist_npy_data. Run [process_mnist.py](mnist_npy_data\process_mnist.py) to create hickle files for subsequent testings.

## Important Data
- Total data: Standard Moving MNIST, 10000 sequences (7000 train, 1000 validation, 2000 test) × 20 frames per sequence
- Data ratios: Train:Validation:Test = 7:1:2
- Total epoch: 40
    - learning rate: 0.001 (first 20 epochs) → 0.0001 (final 20 epochs)
    - Each epoch: all 7000 train sequences + all 1000 validation sequences
- Testing:
    - Frames 0-9 (first 10 frames): provide actual data to model (teacher forcing)
    - Frames 10-19 (last 10 frames): model generates predictions without actual data (autoregressive, self-supervised)
    - Evaluation: separate evaluation code computes MSE/error metrics between predicted frames 10-19 and ground truth




## Transformer Integration
The Transformer module is integrated with PredNet using a **fusion approach**:

- **Components Enhanced**: R (representation) and E (error) layers are enhanced with Transformer (Ahat removed as it has no downstream impact)
- **Spatial Transformer**: Each spatial position in the feature map is treated as a token in a sequence for Transformer processing
- **Output Fusion**: Transformer outputs are blended with PredNet outputs using learnable fusion weights α:
  - `fused_output = σ(α) * transformer_output + (1-σ(α)) * original_output`
  - **Each layer has independent α parameters**: α_E and α_R are learnable per layer (not shared across layers)
  - Each α controls the influence degree: 0.0 (100% original) to 1.0 (100% Transformer)
- **Feedback Mechanism**: The fused R is used in the next layer's ConvLSTM computation and the fused E is used for the next layer's A calculation
- **Benefits**: 
  - Transformer enhances spatial feature representation at each layer
  - Learnable fusion weights allow the model to dynamically balance between original PredNet and Transformer-enhanced features
- **Memory Optimization**:
  - FeedForward dimension: 2× input_dim (reduced from 4× to decrease parameters)
  - Sparse application: Transformer applied every 3 timesteps (not all timesteps)
  - Batch size optimization: 2 for memory efficiency