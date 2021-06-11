
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
import math
from torch.nn.modules.activation import MultiheadAttention
# Import custom modules
from ..transformer.embedding import PatchEmbedding
from ..transformer.layer import TransformerEncoderLayer
from .utils import DiffAugment, DropPath


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

def pixel_upsample(x, H, W): #첫번째 block을 기준으로 했을 때 
    B, N, C = x.size() #N=8(batch_size), S=64(H*W), C=dimention_size 
    assert N == H*W
    x = x.permute(0, 2 ,1) #B(batch_size), C(dimention_size), N(H*W)
    x = x.view(-1, C, H, W) #batch_size, C(dimension_size), H, W 
    x = nn.PixelShuffle(2)(x) # 8, 256, 16, 16 , 2=>upscale factor, (batch_size, dim/4, H*2, W*2) 
    B, C, H, W = x.size() # 8, 256, 16, 16 
    x = x.view(-1, C, H*W) # 8, 256, 16*16
    x = x.permute(0,2,1) #8, 16*16, 256
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
        self.is_mask = is_mask
        self.remove_mask = False
        self.mask_4 = get_attn_mask(is_mask, 4)
        self.mask_5 = get_attn_mask(is_mask, 5)
        self.mask_6 = get_attn_mask(is_mask, 6)
        self.mask_7 = get_attn_mask(is_mask, 7)
        self.mask_8 = get_attn_mask(is_mask, 8)
        self.mask_10 = get_attn_mask(is_mask, 10)

    def forward(self, x, epoch):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) #3, B, self.num_heads, N, C//self.num_heads
        q, k, v = qkv[0], qkv[1], qkv[2]   

        #scaled dot-product attention
        attn = (torch.matmul(q, k.transpose(-2, -1))) * self.scale
        if self.is_mask:
            if epoch < 60:
                if epoch < 22:
                    mask = self.mask_4
                elif epoch < 32:
                    mask = self.mask_6
                elif epoch < 42:
                    mask = self.mask_8
                else:
                    mask = self.mask_10
                attn = attn.masked_fill(mask.to(attn.get_device()) == 0, -1e3) #mask가 0인 tensor 위치에 대해서 -1e3로 채워줌 
            else:
                pass
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = torch.matmul(attn, v).transpose(1, 2).reshape(B, N, C) #attention formula: softmax(self.mat(q, k.transpose(-2,-1))) * head_dim**-0.5) * v 
        x = self.proj(x) #Linear(dim,dim) #need check => 추후에 문서로 남기기 
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    
    def __init__(self, dim=1024, num_heads=4, mlp_ratio=4., drop=0., drop_path=0., is_mask=0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, proj_drop=drop, is_mask=is_mask)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity() #Residual Connection 
        self.norm2 = nn.LayerNorm(dim)
        self.dim_feedforward = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=self.dim_feedforward, drop=drop)

    def forward(self, x, epoch):
        x = x + self.drop_path(self.attn(self.norm1(x), epoch))
        x = x + self.mlp(self.norm2(x))
        return x






class Generator(nn.Module):
    def __init__(self,  latent_dim = 1024, d_model=1024, initial_depth=5, n_head=4, bottom_width=8, drop_path_rate = 0.,
                  dropout=0.1):

        super(Generator, self).__init__()

        self.bottom_width = bottom_width #initial resolution : 8
        self.d_model = d_model
        self.input_linear = nn.Linear(latent_dim, (self.bottom_width ** 2) * self.d_model) #batch size * latent_dim => batch_size * (H*W) * self.d_model 

        # Positional encoding (Learnable positional encoding)
        self.pos_embed_1 = nn.Parameter(torch.zeros(1, self.bottom_width**2,  d_model))  #1, 8*8, 1024
        self.pos_embed_2 = nn.Parameter(torch.zeros(1, (self.bottom_width*2)**2,  d_model//4)) #1, (8*2)*(8*2), (1024/4)=256
        self.pos_embed_3 = nn.Parameter(torch.zeros(1, (self.bottom_width*4)**2,  d_model//16)) #1, (8*4)*(8*4), (1024/16)=64
        self.pos_embed = [
            self.pos_embed_1,
            self.pos_embed_2,
            self.pos_embed_3
        ]

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, int(initial_depth))]

        self.blocks = nn.ModuleList([
            Block(
                dim=d_model, num_heads=n_head, 
                drop=dropout, drop_path=dpr[i])
            for i in range(initial_depth)])

        # Transformer Encoder part
        self.upsample_blocks = nn.ModuleList([
                 nn.ModuleList([
#                     Block(
#                         dim=embed_dim//4, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
#                         drop=drop_rate, attn_drop=attn_drop_rate, drop_path=0, norm_layer=norm_layer),
                    Block(
                        dim=d_model//4, num_heads=n_head, 
                        drop=dropout, is_mask=0),
                    Block(
                        dim=d_model//4, num_heads=n_head, 
                        drop=dropout, is_mask=0),
                    Block(
                        dim=d_model//4, num_heads=n_head, 
                        drop=dropout, is_mask=0),
                    Block(
                        dim=d_model//4, num_heads=n_head, 
                        drop=dropout, is_mask=0)
                 ]
                ),
                 nn.ModuleList([
#                     Block(
#                         dim=embed_dim//16, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
#                         drop=drop_rate, attn_drop=attn_drop_rate, drop_path=0, norm_layer=norm_layer),
                    Block(
                        dim=d_model//16, num_heads=n_head, 
                        drop=dropout, is_mask=0),
                    Block(
                         dim=d_model//16, num_heads=n_head, 
                        drop=dropout, is_mask=(self.bottom_width*4)**2) 
                 ]
                )
                ])
        
        for i in range(len(self.pos_embed)):
            nn.init.trunc_normal_(self.pos_embed[i], std=.02) #표준편차가 .02가 되도록 pos_embed 내 파라미터 값 조정 
        # Deconvolution
        self.deconv = nn.Conv2d(self.d_model//16, 3, 1, 1, 0) #Image Generation 최종 Output=> linear unflatten #왜 Conv2D

        # Initialization


    def forward(self, z, epoch):
        # Noise Input
        x = self.input_linear(z).view(-1, self.bottom_width ** 2, self.d_model) 
        x = x + self.pos_embed[0].to(x.get_device())
        B = x.size()
        H, W = self.bottom_width, self.bottom_width
        for index, blk in enumerate(self.blocks):
            x = blk(x, epoch)
        for index, blk in enumerate(self.upsample_blocks):
            # x = x.permute(0,2,1)
            # x = x.view(-1, self.embed_dim, H, W)
            x, H, W = pixel_upsample(x, H, W)
            x = x + self.pos_embed[index+1].to(x.get_device())
            for b in blk:
                x = b(x, epoch)
            # _, _, H, W = x.size()
            # x = x.view(-1, self.embed_dim, H*W)
            # x = x.permute(0,2,1)
        output = self.deconv(x.permute(0, 2, 1).view(-1, self.d_model//16, H, W))
        return output

