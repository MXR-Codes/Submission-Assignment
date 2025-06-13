import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    """Convert input to a tuple if it's not already one.
    Args:
        t: Input to be converted (either single value or tuple)
    Returns:
        Tuple containing the input value(s)
    """
    return t if isinstance(t, tuple) else (t, t)

# classes

class FeedForward(nn.Module):
    """FeedForward neural network module with GELU activation and dropout."""
    def __init__(self, dim, hidden_dim, dropout = 0.):
        """
        Args:
            dim: Input dimension
            hidden_dim: Hidden layer dimension
            dropout: Dropout probability
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),  # Normalize input
            nn.Linear(dim, hidden_dim),  # First linear layer
            nn.GELU(),  # Gaussian Error Linear Unit activation
            nn.Dropout(dropout),  # Dropout for regularization
            nn.Linear(hidden_dim, dim),  # Second linear layer
            nn.Dropout(dropout)  # Dropout for regularization
        )

    def forward(self, x):
        """Forward pass through the feedforward network."""
        return self.net(x)

class Attention(nn.Module):
    """Multi-head self-attention module."""
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        """
        Args:
            dim: Input dimension
            heads: Number of attention heads
            dim_head: Dimension of each attention head
            dropout: Dropout probability
        """
        super().__init__()
        inner_dim = dim_head *  heads  # Total dimension for all heads
        project_out = not (heads == 1 and dim_head == dim)  # Whether to project output

        self.heads = heads
        self.scale = dim_head ** -0.5  # Scaling factor for dot product attention

        self.norm = nn.LayerNorm(dim)  # Layer normalization

        self.attend = nn.Softmax(dim = -1)  # Softmax for attention weights
        self.dropout = nn.Dropout(dropout)  # Dropout for attention weights

        # Linear layer to generate queries, keys, and values
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        # Output projection layer if needed
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        """Forward pass for self-attention."""
        x = self.norm(x)  # Normalize input

        # Split into queries, keys, and values
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        # Rearrange for multi-head attention
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        # Compute dot product attention
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        # Compute attention weights
        attn = self.attend(dots)
        attn = self.dropout(attn)

        # Apply attention to values
        out = torch.matmul(attn, v)
        # Rearrange back to original shape
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    """Transformer module consisting of multiple attention and feedforward layers."""
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        """
        Args:
            dim: Input dimension
            depth: Number of transformer blocks
            heads: Number of attention heads
            dim_head: Dimension of each attention head
            mlp_dim: Dimension of feedforward hidden layer
            dropout: Dropout probability
        """
        super().__init__()
        self.norm = nn.LayerNorm(dim)  # Final layer normalization
        self.layers = nn.ModuleList([])
        # Create stack of transformer blocks
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        """Forward pass through transformer blocks with residual connections."""
        for attn, ff in self.layers:
            x = attn(x) + x  # Attention with residual connection
            x = ff(x) + x  # Feedforward with residual connection

        return self.norm(x)  # Final normalization

class ViT(nn.Module):
    """Vision Transformer (ViT) model for image classification."""
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        """
        Args:
            image_size: Size of input image (height, width)
            patch_size: Size of each patch (height, width)
            num_classes: Number of output classes
            dim: Embedding dimension
            depth: Number of transformer blocks
            heads: Number of attention heads
            mlp_dim: Dimension of feedforward hidden layer
            pool: Pooling type ('cls' or 'mean')
            channels: Number of input channels
            dim_head: Dimension of each attention head
            dropout: Dropout probability
            emb_dropout: Dropout probability for embeddings
        """
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        # Validate image dimensions are divisible by patch size
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        # Convert image to patch embeddings
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        # Positional embeddings and class token
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        # Transformer encoder
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool  # Pooling method
        self.to_latent = nn.Identity()  # Identity mapping (no operation)

        # Classification head
        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        """Forward pass for Vision Transformer."""
        # Convert image to patch embeddings
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        # Add class token
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        # Add positional embeddings
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        # Pass through transformer
        x = self.transformer(x)

        # Pooling (either use class token or mean pooling)
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)  # Final classification


if __name__ == '__main__':
    # Example usage and testing
    from torchsummary import summary
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Create ViT model instance
    v = ViT(image_size=256,
            patch_size=16,
            num_classes=100,
            dim=1024,
            depth=6,
            heads=16,
            mlp_dim=2048,
            dropout=0.1,
            emb_dropout=0.1)
    v.to(device)
    # Create random input image
    img = torch.randn(1, 3, 256, 256).to(device)
    # summary(v, (3, 256, 256))
    # Get predictions
    preds = v(img)
    print(preds.shape)  # Print output shape