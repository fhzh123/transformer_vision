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
    def __init__(self, in_channels: int = 3, patch_size: int = 16, d_model: int = 768,
                 d_embedding: int = 256, img_size: int = 224):
        super().__init__()
        self.patch_size = patch_size
        self.projection = nn.Sequential(
            # using a conv layer instead of a linear one -> performance gains
            nn.Conv2d(in_channels, d_embedding, kernel_size=patch_size, stride=patch_size),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
        self.cls_token = nn.Parameter(torch.randn(1,1, d_embedding))
        self.embedding_linear = nn.Linear(d_embedding, d_model, bias=False)
        self.positions = nn.Parameter(torch.randn((img_size // patch_size) **2 + 1, d_model))
        
    def forward(self, x: Tensor) -> Tensor:
        batch_size = x.size(0)
        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=batch_size)
        # prepend the cls token to the input
        x = torch.cat([cls_tokens, x], dim=1)
        # Factorized embedding parameterization
        x = self.embedding_linear(x)
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