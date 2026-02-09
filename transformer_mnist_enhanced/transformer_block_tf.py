import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    """
    Single Transformer block with MultiheadAttention and FeedForward.
    Designed to process flattened feature maps of PredNet outputs.
    
    #TODO: Transformer Block - One-layer transformer for E, R, Ahat fusion
    """
    
    def __init__(self, input_dim, num_heads=4, ff_dim=None):
        """
        Arguments:
        -----------
        input_dim: int
            Dimension of input features (flattened spatial dimension)
        num_heads: int
            Number of attention heads (default 4)
        ff_dim: int
            Dimension of feedforward intermediate layer (if None, default to 4*input_dim)
        """
        super(TransformerBlock, self).__init__()
        
        self.input_dim = input_dim
        self.num_heads = num_heads
        #TODO: Transformer Block - Reduced FeedForward dimension from 4x to 2x to save memory
        self.ff_dim = ff_dim if ff_dim is not None else 2 * input_dim
        
        # Ensure input_dim is divisible by num_heads
        assert input_dim % num_heads == 0, f"input_dim ({input_dim}) must be divisible by num_heads ({num_heads})"
        
        # #TODO: Transformer Block - MultiheadAttention layer
        self.attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # #TODO: Transformer Block - Layer normalization after attention
        self.norm1 = nn.LayerNorm(input_dim)
        
        # #TODO: Transformer Block - FeedForward network (expand -> ReLU -> contract)
        self.ff = nn.Sequential(
            nn.Linear(input_dim, self.ff_dim),
            nn.ReLU(),
            nn.Linear(self.ff_dim, input_dim)
        )
        
        # #TODO: Transformer Block - Layer normalization after feedforward
        self.norm2 = nn.LayerNorm(input_dim)
    
    def forward(self, x):
        """
        Forward pass through one transformer block.
        
        Arguments:
        -----------
        x: Tensor
            Input tensor of shape (batch_size, seq_len, input_dim)
            or (batch_size, input_dim) - will be unsqueezed to (batch_size, 1, input_dim)
        
        Returns:
        --------
        output: Tensor
            Output tensor of same shape as input
        """
        # Handle 2D input (batch_size, input_dim) by adding sequence dimension
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch_size, 1, input_dim)
        
        # #TODO: Transformer Block - Self-attention with residual connection
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_output)
        
        # #TODO: Transformer Block - FeedForward with residual connection
        ff_output = self.ff(x)
        x = self.norm2(x + ff_output)
        
        return x
