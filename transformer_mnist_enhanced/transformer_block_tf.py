import torch
import torch.nn as nn
import math

class TransformerBlock(nn.Module):
    """
    Single Transformer block with MultiheadAttention and FeedForward.
    Designed to process flattened feature maps of PredNet outputs.
    
    #TODO: Transformation Enhanced - Added Positional Encodings to restore spatial awareness
    
    #TODO: Transformer Block - One-layer transformer for E, R, Ahat fusion
    """
    
    def __init__(self, input_dim, height=None, width=None, num_heads=4, ff_dim=None, spatial_bias=True):
        """
        Arguments:
        -----------
        input_dim: int
            Dimension of input features (flattened spatial dimension)
        height: int
            Spatial height of the feature map (required for Positional Encoding)
        width: int
            Spatial width of the feature map (required for Positional Encoding)
        num_heads: int
            Number of attention heads (default 4)
        ff_dim: int
            Dimension of feedforward intermediate layer (if None, default to 4*input_dim)
        spatial_bias: bool
            Whether to enforce spatial locality via query masking (default True)
        """
        super(TransformerBlock, self).__init__()
        
        self.input_dim = input_dim
        self.height = height
        self.width = width
        self.spatial_bias = spatial_bias
        self.num_heads = num_heads
        #TODO: Transformer Block - Reduced FeedForward dimension from 4x to 2x to save memory
        self.ff_dim = ff_dim if ff_dim is not None else 2 * input_dim
        
        # Ensure input_dim is divisible by num_heads
        assert input_dim % num_heads == 0, f"input_dim ({input_dim}) must be divisible by num_heads ({num_heads})"
        
        # #TODO: Transformation Enhanced - Helper to create sinusoidal positional encodings
        if self.height is not None and self.width is not None:
            self.pos_encoding = self._create_positional_encoding(self.height, self.width, self.input_dim)
            if self.spatial_bias:
                self.dist_mask = self._create_distance_mask(self.height, self.width)
            else:
                self.dist_mask = None
        else:
            self.pos_encoding = None
            self.dist_mask = None

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

    def _create_positional_encoding(self, h, w, d_model):
        """
        Create 2D sinusoidal positional encoding.
        """
        # We split d_model into two halves: one for y, one for x
        d_y = d_model // 2
        d_x = d_model - d_y

        y_pos = torch.arange(h).unsqueeze(1)
        x_pos = torch.arange(w).unsqueeze(1)

        div_term_y = torch.exp(torch.arange(0, d_y, 2) * (-math.log(10000.0) / d_y))
        div_term_x = torch.exp(torch.arange(0, d_x, 2) * (-math.log(10000.0) / d_x))

        pe_y = torch.zeros(h, d_y)
        pe_y[:, 0::2] = torch.sin(y_pos * div_term_y)
        pe_y[:, 1::2] = torch.cos(y_pos * div_term_y)

        pe_x = torch.zeros(w, d_x)
        pe_x[:, 0::2] = torch.sin(x_pos * div_term_x)
        pe_x[:, 1::2] = torch.cos(x_pos * div_term_x)

        # Broadcast to grid
        pe_y = pe_y.unsqueeze(1).repeat(1, w, 1).view(h * w, d_y)
        pe_x = pe_x.unsqueeze(0).repeat(h, 1, 1).view(h * w, d_x)
        
        pe_full = torch.cat([pe_y, pe_x], dim=1)
        return nn.Parameter(pe_full.unsqueeze(0), requires_grad=False)

    def _create_distance_mask(self, h, w):
        """
        Create a distance-based attention bias mask.
        Decays attention scores based on squared Euclidean distance.
        result[i, j] ~ -distance(i, j)^2
        """
        # Grid coordinates
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        coords = torch.stack([y.flatten(), x.flatten()], dim=1).float()
        
        # Pairwise squared Euclidean distance
        # (N, 2) -> (N, N) distance matrix
        dist_sq = torch.cdist(coords, coords, p=2) ** 2
        
        # Normalize/Scale: We want distant pixels to have large negative values
        # Sigma controls the "width" of the focus. Let's say sigma = min(h, w) / 2
        sigma = min(h, w) / 2.0
        mask = -dist_sq / (2 * (sigma ** 2))
        
        # This mask will be added to softmax logits. 
        # Large distance -> Large negative value -> prob -> 0.
        return nn.Parameter(mask, requires_grad=False)
    
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
        
        bs, seq_len, _ = x.shape
        
        # #TODO: Transformation Enhanced - Add positional encoding
        if self.pos_encoding is not None:
             # Ensure PE is on same device
            if self.pos_encoding.device != x.device:
                self.pos_encoding = self.pos_encoding.to(x.device)
                if self.dist_mask is not None:
                    self.dist_mask = self.dist_mask.to(x.device)
            
            # Slice in case sequence length differs (though it shouldn't for fixed image size)
            x = x + self.pos_encoding[:, :seq_len, :]

        # #TODO: Transformer Block - Self-attention with residual connection
        # attn_mask expects (seq_len, seq_len)
        mask = self.dist_mask if (self.spatial_bias and self.dist_mask is not None) else None
        
        attn_output, _ = self.attention(x, x, x, attn_mask=mask)
        x = self.norm1(x + attn_output)
        
        # #TODO: Transformer Block - FeedForward with residual connection
        ff_output = self.ff(x)
        x = self.norm2(x + ff_output)
        
        return x
