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

class Discriminator(nn.Module):
    def __init__(self, args, img_size=32, patch_size=4, in_chans=3, num_classes=1, embed_dim=None, depth=7,
                 num_heads=4, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = embed_dim = self.embed_dim = args.df_dim  # num_features for consistency with other models
        depth = 7
        self.args = args

        patch_size = 4

        patch_size = args.patch_size


        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size, padding=0)
        num_patches = (args.img_size // patch_size)**2

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            D_Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # NOTE as per official impl, we could have a pre-logits representation dense layer + tanh here
        #self.repr = nn.Linear(embed_dim, representation_size)
        #self.repr_act = nn.Tanh()

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x).flatten(2).permute(0,2,1)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x[:,0]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x
