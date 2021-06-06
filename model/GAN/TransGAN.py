import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from torch.nn.modules.activation import MultiheadAttention
# Import custom modules
from ..transformer.embedding import PatchEmbedding
from ..transformer.layer import TransformerEncoderLayer
from utils import DiffAugment


class matmul(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x1, x2):
        x = x1@x2
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def pixel_upsample(x, H, W):
    N, S, C = x.size()
    assert S == H*W
    x = x.permute(1, 0, 2).contiguous()
    x = x.view(-1, C, H, W)
    x = nn.PixelShuffle(2)(x)
    _, C, H, W = x.size()
    x = x.view(-1, C, H*W)
    x = x.permute(0,2,1)
    return x, H, W

def get_attn_mask(N, w):
    mask = torch.zeros(1, 1, N, N).cuda()
    for i in range(N):
        if i <= w:
            mask[:, :, i, 0:i+w+1] = 1
        elif N - i <= w:
            mask[:, :, i, i-w:N] = 1
        else:
            mask[:, :, i, i:i+w+1] = 1
            mask[:, :, i, i-w:i] = 1
    return mask

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., is_mask=0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.mat = matmul()
        self.is_mask = is_mask
        self.remove_mask = False
        self.mask_4 = get_attn_mask(is_mask, 8)
        self.mask_5 = get_attn_mask(is_mask, 10)
        self.mask_6 = get_attn_mask(is_mask, 12)
        self.mask_7 = get_attn_mask(is_mask, 14)
        self.mask_8 = get_attn_mask(is_mask, 16)
        self.mask_10 = get_attn_mask(is_mask, 20)

    def forward(self, x, epoch):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (self.mat(q, k.transpose(-2, -1))) * self.scale
        if self.is_mask:
            if epoch < 20:
                if epoch < 5:
                    mask = self.mask_4
                elif epoch < 10:
                    mask = self.mask_6
                elif epoch < 15:
                    mask = self.mask_8
                else:
                    mask = self.mask_10
                attn = attn.masked_fill(mask.to(attn.get_device()) == 0, -1e3)
            else:
                pass
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = self.mat(attn, v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    
    def __init__(self, dim=1024, num_heads=4, drop=0., dim_feedforward = 4096, is_mask=0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, proj_drop=drop, is_mask=is_mask)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.norm2 = nn.LayerNorm(dim)
        
        self.mlp = Mlp(in_features=dim, hidden_features=dim_feedforward, drop=drop)

    def forward(self, x, epoch):
        x = x + self.attn(self.norm1(x), epoch)
        x = x + self.mlp(self.norm2(x))
        return x

class Generator(nn.Module):
    def __init__(self,  latent_dim = 1024, d_model=1024, depth='542', n_head=4, bottom_width=8, 
                 dim_feedforward=512, dropout=0.1):

        super(Generator, self).__init__()

        self.bottom_width = bottom_width
        self.d_model = d_model
        self.input_linear = nn.Linear(latent_dim, (self.bottom_width ** 2) * self.d_model)

        # Position Embedding
        self.pos_embed_1 = nn.Parameter(torch.zeros(1, self.bottom_width**2,  d_model))
        self.pos_embed_2 = nn.Parameter(torch.zeros(1, (self.bottom_width*2)**2,  d_model//4))
        self.pos_embed_3 = nn.Parameter(torch.zeros(1, (self.bottom_width*4)**2,  d_model//16))
        self.pos_embed = [
            self.pos_embed_1,
            self.pos_embed_2,
            self.pos_embed_3
        ]

        # Transformer Encoder part
        self.transformer_blocks = nn.ModuleList([
                nn.ModuleList([
                     Block(
                    dim=d_model, num_heads=n_head, drop=dropout, dim_feedforward = dim_feedforward, is_mask = 0 )
            for i in range(int(depth[0]))]),
                nn.ModuleList([
                    Block(
                    dim=d_model//4, num_heads=n_head, drop=dropout, dim_feedforward = dim_feedforward, is_mask = 0)
                        for i in range(int(depth[1]))]),
                nn.ModuleList([
                    Block(
                    dim=d_model//16, num_heads=n_head, drop=dropout, dim_feedforward = dim_feedforward, is_mask = (self.bottom_width*4)**2)
                    if i == int(depth[2])-1
                    
                    else
                        Block(
                        dim=d_model//16, num_heads=n_head, drop=dropout, dim_feedforward = dim_feedforward, is_mask = 0 ) 
                    for i in range(int(depth[2]))])
                    ])
        
        for i in range(len(self.pos_embed)):
            nn.init.trunc_normal_(self.pos_embed[i], std=.02)
        # Deconvolution
        self.deconv = nn.Conv2d(self.d_model//16, 3, 1, 1, 0)

        # Initialization


    def forward(self, z, epoch):
        # Noise Input
        x = self.input_linear(z).view(-1, self.bottom_width ** 2, self.d_model)
        H, W = self.bottom_width, self.bottom_width
        # Transformer encoder
        for i, blocks in enumerate(self.transformer_blocks):
            if i == 0:
                for encoder in blocks:
                    x += self.pos_embed[i]
                    x = encoder(x, epoch)
            else:
                x, H, W = pixel_upsample(x, H, W)
                x += self.pos_embed[i]
                for encoder in blocks:
                    x = encoder(x, epoch)
        # De-convolution
        output = self.deconv(x.permute(0, 2, 1).reshape(-1, self.d_model//16, H, W))
        return output

class Discriminator(nn.Module):
    def __init__(self,  n_classes: int, d_model: int = 512, d_embedding: int = 256, 
                 n_head: int = 8, dim_feedforward: int = 2048,
                 num_encoder_layer: int = 10, img_size: int = 224, patch_size: int = 16, 
                 dropout: float = 0.3, triple_patch: bool = False, diff_aug: str = "translation,cutout,color" ):
    
        super(Discriminator, self).__init__()

        self.diff_aug = diff_aug

        self.dropout = nn.Dropout(dropout)

        self.patch_embed = nn.Conv2d(3, d_model, kernel_size=patch_size, stride=patch_size, padding=0)
        num_patches = (img_size // patch_size)**2

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, d_model))

        self.encoders = nn.ModuleList([
           Block(
                dim=d_model, num_heads=n_head, drop=dropout, dim_feedforward = dim_feedforward)
                for i in range(num_encoder_layer)])

        self.norm = nn.LayerNorm(d_model)

        self.head = nn.Linear(d_model, n_classes) if n_classes > 0 else nn.Identity()


    def forward(self,  src_img: Tensor, epoch) -> Tensor:
        
        if self.diff_aug!="None":
            src_img = DiffAugment(src_img, self.diff_aug, True)

        B = src_img.shape[0]
        x = self.patch_embed(src_img).flatten(2).permute(0, 2, 1)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x ), dim=1)
        x = x + self.pos_embed

        # Transformer Encoder
        for encoder in self.encoders:
            encoder_out = encoder(x, epoch)
        
        encoder_out = self.norm(encoder_out)

        encoder_out= self.head(encoder_out[:,0])

        return encoder_out
