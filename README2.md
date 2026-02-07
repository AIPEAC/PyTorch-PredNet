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

## Transformer Integration Strategy
The Transformer module is integrated with PredNet using a **fusion and feedback approach**:

- **Input Processing**: At each timestep, after PredNet produces predictions, all layer outputs (E, R, Ahat) are fed independently into the Transformer as a sequence
- **Temporal Modeling**: The Transformer maintains its hidden state across timesteps to capture temporal dependencies and patterns in the prediction sequence
- **Output Fusion**: Transformer outputs are blended with PredNet outputs using a learnable or fixed fusion weight α:
  - `combined_output = α * transformer_output + (1-α) * prednet_output`
  - α controls the influence degree: 0% (pure PredNet) to 100% (pure Transformer)
- **Feedback Loop**: The fused outputs serve as the next timestep's E/R/Ahat inputs for PredNet, allowing the Transformer to progressively refine predictions while maintaining PredNet's representational stability
- **Benefits**: 
  - Transformer learns long-term temporal patterns across the prediction sequence
  - Incremental refinement: combines PredNet's stability with Transformer's adaptive correction capability
  - Memory preservation: Transformer's hidden state enables cross-timestep learning and pattern recognition

