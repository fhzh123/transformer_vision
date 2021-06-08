# Import modules
import math
from einops import repeat
from einops.layers.torch import Rearrange
# Import PyTorch
import torch
from torch import nn
from torch import Tensor
from torch.nn import functional as F
from torch.cuda.amp.autocast_mode import autocast

class PatchEmbedding(nn.Module):
    """
    Embedding which is consisted with under features
    1. projection : using conv layer to flatten and rearrange
    2. positions : adding positional information using parameters
    sum of all these features are output of Embedding
    then use Factorized embedding parameterization from ALBERT (Z Lan et al. 2019)
    """
    def __init__(self, in_channels: int = 3, patch_size: int = 16, d_model: int = 768,
                 img_size: int = 224, triple_patch: bool = False):
        super().__init__()
        # Hyper-parameter setting
        self.patch_size = patch_size
        self.triple_patch = triple_patch

        # Patch projection & parameter setting
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size),
            Rearrange('b e (h) (w) -> b (h w) e')
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.positions = nn.Parameter(torch.randn((img_size // patch_size)**2 + 1, d_model))
        
        # For triple patch
        if self.triple_patch:
            self.projection_half = nn.Sequential(
                nn.Conv2d(in_channels, d_model, kernel_size=patch_size//2, stride=patch_size//2),
                Rearrange('b e (h) (w) -> b (h w) e')
            )
            self.positions_half = nn.Parameter(torch.randn((img_size // (patch_size//2))**2 + 1, d_model))
            self.projection_double = nn.Sequential(
                nn.Conv2d(in_channels, d_model, kernel_size=patch_size*2, stride=patch_size*2),
                Rearrange('b e (h) (w) -> b (h w) e')
            )
            self.positions_double = nn.Parameter(torch.randn((img_size // (patch_size*2))**2 + 1, d_model))

            self.seg_token = nn.Parameter(torch.randn(1, 1, d_model))
            self.seg_original = nn.Parameter(torch.randn(1, d_model))
            self.seg_half = nn.Parameter(torch.randn(1, d_model))
            self.seg_double = nn.Parameter(torch.randn(1, d_model))

    @autocast()
    def forward(self, x: Tensor) -> Tensor:
        # prepare settings
        batch_size = x.size(0)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=batch_size)
        # project original patch
        x_original = self.projection(x)
        # prepend the cls token to the input
        x_original = torch.cat([cls_tokens, x_original], dim=1)
        # add position embedding
        x_original += self.positions
        # triple patch mode
        if self.triple_patch:
            x_original += self.seg_original
            # prepare segment token
            seg_tokens = repeat(self.seg_token, '() n e -> b n e', b=batch_size)
            # project half size patch
            x_half = self.projection_half(x)
            x_half = torch.cat([seg_tokens, x_half], dim=1)
            x_half += self.positions_half 
            x_half += repeat(self.seg_half, '() e -> n e', n=x_half.size(1))
            # project double size patch
            x_double = self.projection_double(x)
            x_double = torch.cat([seg_tokens, x_double], dim=1)
            x_double += self.positions_double
            x_double += repeat(self.seg_double, '() e -> n e', n=x_double.size(1))
            # concatenate
            x_out = torch.cat([x_original, x_half, x_double], dim=1)
        else:
            x_out = x_original

        return x_out

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=512):
        super().__init__()

        pe = torch.zeros(max_len, d_model, dtype=torch.float)
        pe.require_grad = False

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2, dtype=torch.float) * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    @autocast()
    def forward(self, x):
        return self.pe[:, :x.size(1)]

class TransformerEmbedding(nn.Module):
    """
    Embedding which is consisted with under features
    1. TokenEmbedding : normal embedding matrix
    2. PositionalEncoding : adding positional information using sin, cos
    sum of all these features are output of Embedding
    """
    def __init__(self, vocab_size, d_model, embed_size, pad_idx=0, max_len=512, embedding_dropout=0.1):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()
        self.token = nn.Embedding(vocab_size, embed_size, padding_idx=pad_idx)
        self.linear_layer = nn.Linear(embed_size, d_model)
        self.position = PositionalEncoding(d_model=d_model, max_len=max_len)
        self.embed_norm = nn.LayerNorm(d_model, eps=1e-12)
        self.dropout = nn.Dropout(embedding_dropout)

    @autocast()
    def forward(self, sequence):
        x = self.dropout(F.gelu(self.linear_layer(self.token(sequence))))
        x = self.embed_norm(x + self.position(sequence))
        return x