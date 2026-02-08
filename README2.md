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
- Total epoch: 150
    - learning rate: 0.001 (first 75 epochs) → 0.0001 (final 75 epochs)
    - Each epoch: all 7000 train sequences + all 1000 validation sequences
- Testing:
    - Frames 0-9 (first 10 frames): provide actual data to model (teacher forcing)
    - Frames 10-19 (last 10 frames): model generates predictions without actual data (autoregressive, self-supervised)
    - Evaluation: separate evaluation code computes MSE/error metrics between predicted frames 10-19 and ground truth




## Transformer Integration
The Transformer module is integrated with PredNet using a **fusion and feedback approach**:

- **Input Processing**: At each timestep, after PredNet produces predictions, all layer outputs (E, R, Ahat) are fed independently into the Transformer as a sequence
- **Temporal Modeling**: The Transformer maintains its hidden state across timesteps to capture temporal dependencies and patterns in the prediction sequence
- **Output Fusion**: Transformer outputs are blended with PredNet outputs using learnable fusion weights α:
  - `combined_output = α * transformer_output + (1-α) * prednet_output`
  - **Each layer has independent α parameters**: α_E, α_R, α_Ahat are learnable per layer (not shared across layers)
  - Each α controls the influence degree: 0% (pure PredNet) to 100% (pure Transformer)
- **Feedback Loop**: The fused outputs serve as the next timestep's E/R/Ahat inputs for PredNet, allowing the Transformer to progressively refine predictions while maintaining PredNet's representational stability
- **Benefits**: 
  - Transformer learns long-term temporal patterns across the prediction sequence
  - Incremental refinement: combines PredNet's stability with Transformer's adaptive correction capability
  - Memory preservation: Transformer's hidden state enables cross-timestep learning and pattern recognition
- **FeedForward Dimension**: 
  - Reduced from 4× to 2× input_dim to decrease parameter count and memory usage
  - Previous: Linear(input_dim → 4×input_dim → input_dim)
  - Current: Linear(input_dim → 2×input_dim → input_dim)