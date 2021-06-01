import torch
import torch.nn as nn
from torch.nn.modules.activation import MultiheadAttention
from ..transformer.layer import TransformerEncoderLayer

def pixel_upsample(x, H, W):
    _, N, C = x.size()
    assert N == H*W
    x = x.permute(0, 2, 1)
    x = x.view(-1, C, H, W)
    x = nn.PixelShuffle(2)(x)
    _, C, H, W = x.size()
    x = x.view(-1, C, H*W)
    x = x.permute(0,2,1)
    return x, H, W

class Generator(nn.Module):
    def __init__(self, d_model=384, num_encoder_layer=5, n_head=4, bottom_width=4, 
                 dim_feedforward=1024, dropout=0.1):
        super(Generator, self).__init__()
        self.bottom_width = bottom_width
        self.d_model = d_model
        self.input_linear = nn.Linear(args.latent_dim, (self.bottom_width ** 2) * self.d_model)
        # Position Embedding
        self.pos_embed_1 = nn.Parameter(torch.randn(1, self.bottom_width**2, d_model))
        self.pos_embed_2 = nn.Parameter(torch.randn(1, (self.bottom_width*2)**2, d_model//4))
        self.pos_embed_3 = nn.Parameter(torch.randn(1, (self.bottom_width*4)**2, d_model//16))
        self.pos_embed_4 = nn.Parameter(torch.randn(1, (self.bottom_width*8)**2, d_model//64))
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
                        d_model, n_head, dropout=dropout), dim_feedforward, dropout=dropout) \
                        for i in range(num_encoder_layer)]),
                nn.ModuleList([
                    TransformerEncoderLayer(d_model//16, MultiheadAttention(
                        d_model, n_head, dropout=dropout), dim_feedforward, dropout=dropout) \
                        for i in range(num_encoder_layer)]),
                nn.ModuleList([
                    TransformerEncoderLayer(d_model//64, MultiheadAttention(
                        d_model, n_head, dropout=dropout), dim_feedforward, dropout=dropout) \
                        for i in range(num_encoder_layer)])
        ])
        # Deconvolution
        self.deconv = nn.Conv2d(self.d_model//64, 3, 1, 1, 0)

    def forward(self, z, epoch):
        # Noise Input
        x = self.input_linear(z).view(-1, self.bottom_width ** 2, self.d_model)
        x = x + self.pos_embed[0].to(x.get_device())
        H, W = self.bottom_width, self.bottom_width
        # Transformer encoder
        for i, blocks in enumerate(self.transformer_blocks):
            if i == 0:
                for encoder in blocks:
                    x = encoder(x)
            else:
                x, H, W = pixel_upsample(x, H, W)
                x = x + self.pos_embed[i].to(x.get_device())
                for b in blocks:
                    x = b(x, epoch)
        # De-convolution
        output = self.deconv(x.permute(0, 2, 1).view(-1, self.d_model//64, H, W))
        return output