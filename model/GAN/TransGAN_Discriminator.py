import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from torch.nn.modules.activation import MultiheadAttention
# Import custom modules
from ..transformer.embedding import PatchEmbedding
from .utils import DiffAugment, DropPath, Mlp



class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (torch.matmul(q, k.transpose(-2, -1))) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = torch.matmul(attn, v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Discriminator_block(nn.Module):
    
    def __init__(self, dim=None, attn=None, num_heads=4, mlp_ratio=4., drop=0., drop_path=0., is_mask=0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity() #need to check
        self.dim_feedforward = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=self.dim_feedforward, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm(x)))
        x = x + self.drop_path(self.mlp(self.norm(x)))
        return x


class Discriminator(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, n_classes=1, d_model=None, n_head=4, num_encoder_layer=7, img_size=32, patch_size=4, dropout=0., 
                  diff_aug='translation,cutout,color'):
        super(Discriminator, self).__init__()
        
        self.pos_drop = nn.Dropout(p=dropout)
        self.diff_aug_type = diff_aug
       

        #Image embedding part
        self.patch_embedding = PatchEmbedding(in_channels=3, patch_size=patch_size, d_model=d_model, img_size=img_size, triple_patch=False)

        #Transformer Encoder part
        dpr = [x.item() for x in torch.linspace(0, dropout, num_encoder_layer)]   # stochastic depth decay rule - dropout=0인 경우 어떠한 경로든 drop_path 적용 X 
        self.blocks = nn.ModuleList([
            Discriminator_block(dim=d_model, num_heads=n_head, drop=dropout, drop_path=dpr[i])
            for i in range(num_encoder_layer)])

        self.norm = nn.LayerNorm(d_model)

        # Classifier head
        self.head = nn.Linear(d_model, n_classes) if n_classes > 0 else nn.Identity()
        
    def forward_features(self, x):
        
        if "None" not in self.diff_aug_type:
             diff_aug_cls = DiffAugment(x, self.diff_aug_type, True)
             x = diff_aug_cls.diff_augment()
        x = self.patch_embedding(x)
        
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x[:,0]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


