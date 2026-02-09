# Transformer Enhanced MNIST Project

Based on the original PredNet Pytorch implementation and the Transformer MNIST project.
This version enhances the Transformer module by adding "Spatial Awareness".

## Improvement: Spatial Awareness & Positional Encoding

**The Problem:**
In the original Transformer implementation (`transformer_mnist`), the 2D feature maps from PredNet were simply flattened into a 1D sequence before being fed into the Transformer.
Since Transformers are naturally permutation-invariant (they don't know the order of elements), the model lost all spatial context. It couldn't distinguish between a pixel at (0,0) and a pixel at (10,10). The spatial structure crucial for image prediction was destroyed, causing the model to down-weight the Transformer output (alpha -> 0).

**The Solution (`transformer_mnist_enhanced`):**
We introduced two mechanisms to restore spatial sense:

1.  **2D Sinusoidal Positional Encoding**: 
    -   We inject a fixed position signature into each pixel's feature vector.
    -   Pixel $(x, y)$ gets a unique embedding vector $P_{(x,y)}$.
    -   This allows the Transformer to mathematically distinguish relative and absolute positions. 
    -   *Analogy*: We gave the "blind" Transformer a map of the classroom, so it knows who sits next to whom.

2.  **Distance-based Attention Bias (The "Square Function"):**
    -   As requested, we added a penalty term to the Attention mechanism based on physical distance.
    -   $\text{Mask}[i, j] \propto - \frac{\text{Distance}(i, j)^2}{\sigma^2}$
    -   This forces the attention mechanism to prioritize "nearby" pixels (tight connection) and exponentially decay the connection strength for distant pixels.
    -   *Analogy*: We made the "voice" of distant students softer, so pixels mostly listen to their neighbors, preserving the local structure that PredNet loves.

## How to Run

Navigate to `transformer_mnist_enhanced/` and run the training script.

```bash
cd transformer_mnist_enhanced
python mnist_train_all_sparse.py
```

## Files Changed from `transformer_mnist`

-   **`transformer_block_tf.py`**:
    -   Added `PositionalEncoding2D` class.
    -   Added `_create_distance_mask` method for the "square function" decay.
    -   Modified `forward` to add positional embeddings and apply the distance mask.
-   **`prednet_tf_sparse.py`**:
    -   Modified `__init__` to pass `height` and `width` to `TransformerBlock`.

All other files (`mnist_train_all_sparse.py`, etc.) are preserved to maintain the sparse training configuration.
