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
                 img_size: int = 224, triple_patch: bool = False, device: torch.device = None):
        super().__init__()
        self.patch_size = patch_size
        self.triple_patch = triple_patch
        self.device = device
        if self.triple_patch:
            self.projection = {
                0: nn.Sequential(
                # using a conv layer instead of a linear one -> performance gains
                nn.Conv2d(in_channels, d_model, kernel_size=patch_size//2, stride=patch_size//2),
                Rearrange('b e (h) (w) -> b (h w) e')
                ).to(device),
                1: nn.Sequential(
                # using a conv layer instead of a linear one -> performance gains
                nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size),
                Rearrange('b e (h) (w) -> b (h w) e')
                ).to(device),
                2: nn.Sequential(
                # using a conv layer instead of a linear one -> performance gains
                nn.Conv2d(in_channels, d_model, kernel_size=patch_size*2, stride=patch_size*2),
                Rearrange('b e (h) (w) -> b (h w) e')
                ).to(device)
            }
            self.segment_embedding = nn.Embedding(3, d_model)
            self.positions = {
                0: nn.Parameter(torch.randn((img_size // (patch_size//2)) **2 + 1, d_model)).to(device),
                1: nn.Parameter(torch.randn((img_size // patch_size) **2 + 1, d_model)).to(device),
                2: nn.Parameter(torch.randn((img_size // (patch_size*2)) **2 + 1, d_model)).to(device)
            }
        else:
            self.projection = nn.Sequential(
                # using a conv layer instead of a linear one -> performance gains
                nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size),
                Rearrange('b e (h) (w) -> b (h w) e'),
            )
            self.positions = nn.Parameter(torch.randn((img_size // patch_size) **2 + 1, d_model))
        self.cls_token = nn.Parameter(torch.randn(1,1, d_model))
        
    def forward(self, x: Tensor) -> Tensor:
        batch_size = x.size(0)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=batch_size)
        if self.triple_patch:
            # Triple batch projection
            x0 = self.projection[0](x)
            x0 = torch.cat([cls_tokens, x0], dim=1)
            x0 += self.positions[0]
            x0_seg = repeat(self.segment_embedding(torch.cuda.LongTensor([[0]])), 
                '() () e -> b n e', b=batch_size, n=x0.size(1))
            x1 = self.projection[1](x)
            x1 = torch.cat([cls_tokens, x1], dim=1)
            x1 += self.positions[1]
            x1_seg = repeat(self.segment_embedding(torch.cuda.LongTensor([[1]])), 
                '() () e -> b n e', b=batch_size, n=x1.size(1))
            x2 = self.projection[2](x)
            x2 = torch.cat([cls_tokens, x2], dim=1)
            x2 += self.positions[2]
            x2_seg = repeat(self.segment_embedding(torch.cuda.LongTensor([[2]])), 
                '() () e -> b n e', b=batch_size, n=x2.size(1))
            # concat triple patch tensor
            x = torch.cat([x0, x1, x2], dim=1)
            # add segment embedding
            x += torch.cat([x0_seg, x1_seg, x2_seg], dim=1)
        else:
            x = self.projection(x)
            # prepend the cls token to the input
            x = torch.cat([cls_tokens, x], dim=1)
            # add position embedding
            x += self.positions

        return x

class PositionalEmbedding(nn.Module):

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
    2. PositionalEmbedding : adding positional information using sin, cos
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
        self.position = PositionalEmbedding(d_model=d_model, max_len=max_len)
        self.embed_norm = nn.LayerNorm(d_model, eps=1e-12)
        self.dropout = nn.Dropout(embedding_dropout)

    def forward(self, sequence):
        x = self.dropout(F.gelu(self.linear_layer(self.token(sequence))))
        x = self.embed_norm(x + self.position(sequence))
        return x