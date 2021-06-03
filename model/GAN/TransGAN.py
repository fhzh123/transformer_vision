import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from torch.nn.modules.activation import MultiheadAttention
# Import custom modules
from ..transformer.embedding import PatchEmbedding
from ..transformer.layer import TransformerEncoderLayer

def pixel_upsample(x, H, W):
    N, _, C = x.size()
    assert N == H*W
    x = x.permute(1, 0, 2).contiguous()
    x = x.view(-1, C, H, W)
    x = nn.PixelShuffle(2)(x)
    _, C, H, W = x.size()
    x = x.view(-1, C, H*W)
    x = x.permute(2,0,1)
    return x, H, W

class Generator(nn.Module):
    def __init__(self, d_model=256, num_encoder_layer=5, n_head=4, bottom_width=8, 
                 dim_feedforward=512, dropout=0.1):

        super(Generator, self).__init__()

        self.bottom_width = bottom_width
        self.d_model = d_model
        self.input_linear = nn.Linear(256, (self.bottom_width ** 2) * self.d_model)

        # Position Embedding
        self.pos_embed_1 = nn.Parameter(torch.randn(self.bottom_width**2, 1, d_model))
        self.pos_embed_2 = nn.Parameter(torch.randn((self.bottom_width*2)**2, 1, d_model//4))
        self.pos_embed_3 = nn.Parameter(torch.randn((self.bottom_width*4)**2, 1, d_model//16))
        self.pos_embed_4 = nn.Parameter(torch.randn((self.bottom_width*8)**2, 1, d_model//64))
        self.pos_embed = [
            self.pos_embed_1,
            self.pos_embed_2,
            self.pos_embed_3,
            self.pos_embed_4
        ]

        # Transformer Encoder part
        self.transformer_blocks = nn.ModuleList([
                nn.ModuleList([
                    TransformerEncoderLayer(d_model, MultiheadAttention(
                        d_model, n_head, dropout=dropout), dim_feedforward, dropout=dropout) \
                        for i in range(num_encoder_layer)]),
                nn.ModuleList([
                    TransformerEncoderLayer(d_model//4, MultiheadAttention(
                        d_model//4, n_head, dropout=dropout), dim_feedforward, dropout=dropout) \
                        for i in range(num_encoder_layer)]),
                nn.ModuleList([
                    TransformerEncoderLayer(d_model//16, MultiheadAttention(
                        d_model//16, n_head, dropout=dropout), dim_feedforward, dropout=dropout) \
                        for i in range(num_encoder_layer)]),
                nn.ModuleList([
                    TransformerEncoderLayer(d_model//64, MultiheadAttention(
                        d_model//64, n_head, dropout=dropout), dim_feedforward, dropout=dropout) \
                        for i in range(num_encoder_layer)])
        ])
        # Deconvolution
        self.deconv = nn.Conv2d(self.d_model//64, 3, 1, 1, 0)

        # Initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.kaiming_uniform_(p) 

    def forward(self, z):
        # Noise Input
        x = self.input_linear(z).view(self.bottom_width ** 2, -1, self.d_model)
        x += self.pos_embed[0]
        H, W = self.bottom_width, self.bottom_width
        # Transformer encoder
        for i, blocks in enumerate(self.transformer_blocks):
            if i == 0:
                for encoder in blocks:
                    x = encoder(x)
            else:
                x, H, W = pixel_upsample(x, H, W)
                x += self.pos_embed[i]
                for encoder in blocks:
                    x = encoder(x)
        # De-convolution
        output = self.deconv(x.transpose(0, 1).contiguous().view(-1, self.d_model//64, H, W))
        return output

class Discriminator(nn.Module):
    def __init__(self, n_classes: int, d_model: int = 512, d_embedding: int = 256, 
                 n_head: int = 8, dim_feedforward: int = 2048,
                 num_encoder_layer: int = 10, img_size: int = 224, patch_size: int = 16, 
                 dropout: float = 0.3, triple_patch: bool = False):
    
        super(Discriminator, self).__init__()

        self.dropout = nn.Dropout(dropout)

        # Image embedding part
        self.patch_embedding = PatchEmbedding(in_channels=3, patch_size=patch_size,
            d_model=d_model, img_size=img_size, triple_patch=triple_patch)

        # Transformer Encoder part
        self_attn = MultiheadAttention(d_model, n_head, dropout=dropout)
        self.encoders = nn.ModuleList([
            TransformerEncoderLayer(d_model, self_attn, dim_feedforward, dropout=dropout) \
                for i in range(num_encoder_layer)])

        # Target linear part (Not averaging)
        self.trg_output_linear = nn.Linear(d_model, d_embedding)
        self.trg_output_norm = nn.LayerNorm(d_embedding, eps=1e-12)
        self.trg_output_linear2 = nn.Linear(d_embedding, n_classes)

        # Initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.kaiming_uniform_(p) 

    def forward(self, src_img: Tensor) -> Tensor:
        # Image embedding
        encoder_out = self.patch_embedding(src_img).transpose(0, 1)
        
        # Transformer Encoder
        for encoder in self.encoders:
            encoder_out = encoder(encoder_out)

        # Target linear
        encoder_out = encoder_out.transpose(0, 1)
        encoder_out = self.trg_output_norm(self.dropout(F.gelu(self.trg_output_linear(encoder_out))))
        encoder_out = self.trg_output_linear2(encoder_out)
        return encoder_out
