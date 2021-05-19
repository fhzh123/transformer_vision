# Import PyTorch
from torch.nn.modules.pixelshuffle import PixelShuffle
from model.transformer.embedding import PositionalEmbedding
import torch
from torch import nn
from torch import Tensor
from torch.nn import functional as F
from torch.nn.modules.activation import MultiheadAttention
# Import custom modules
from ..transformer.embedding import PatchEmbedding
from ..transformer.layer import TransformerEncoderLayer
# from .layers import PatchEmbedding
# from .transformer import TransformerEncoder, ClassificationHead

def pixel_upsample(x, H, W):
    B, N, C = x.size()
    assert N == H*W
    x = x.permute(0, 2, 1)
    x = x.view(-1, C, H, W)
    x = nn.PixelShuffle(2)(x)
    B, C, H, W = x.size()
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

class Generator(nn.Module): #Block은 그대로 반영 => Discriminator는 ViT 그대로 사용 
    def __init__(self, args, model_type='TransGAN_XL', bottom_width = 8,  
                 n_head = 8, dim_feedforward: int = 2048, latent_dim = 1024):
        super(Generator, self).__init__()
        self.ch = args.d_embedding
        self.dropout_ratio = args.dropout
        self.bottom_width = bottom_width #MLP 통과 이후 => 이미지의 시작 사이즈 
        self.embed_dim = args.d_embedding
        self.is_mask = 0

        #MLP for noise
        self.mlp = nn.Linear(latent_dim, (self.bottom_width**2) * self.embed_dim)
        
        # Transformer Encoder part
        if model_type == 'TransGAN_XL': 
            self_attn_1 = MultiheadAttention(self.embed_dim, n_head, dropout=self.dropout_ratio)
            self.first_stage_encoders = nn.ModuleList([
                TransformerEncoderLayer(self.embed_dim, self_attn_1, dim_feedforward, dropout=self.dropout_ratio ) \
                for i in range(5)
                ])
            self_attn_2 = MultiheadAttention(self.embed_dim//4, n_head, dropout=self.dropout_ratio)
            self.second_stage_encoders = nn.ModuleList([
                TransformerEncoderLayer(self.embed_dim//4, self_attn_2, dim_feedforward, dropout=self.dropout_ratio ) \
                for i in range(4)
            ])
            self_attn_3 = MultiheadAttention(self.embed_dim//16, n_head, dropout=self.dropout_ratio)
            self.third_stage_encoders = nn.ModuleList([
                TransformerEncoderLayer(self.embed_dim//16, self_attn_3, dim_feedforward, dropout=self.dropout_ratio ) \
                for i in range(2)
            ])
            self.pos_embed_1 = PositionalEmbedding(d_model = self.embed_dim)

            self.deconv = nn.Sequential(
            nn.Conv2d(self.embed_dim//16, 3, 1, 1, 0)
            )

            self.to_rgb = nn.Sequential(
                nn.BatchNorm2d(args.latent_dim),
                nn.ReLU(),
                # nn.Conv2d(args.gf_dim, 3, 3, 1, 1),
                nn.Tanh()
                )

    def forward(self, z):
        
        x = self.mlp(z).view(-1, self.bottom_width ** 2, self.embed_dim)
        x = x + self.pos_embed_1(x)
        H, W= self.bottom_width, self.bottom_width

        for first_stage_encoder in self.first_stage_encoders: 
            x = first_stage_encoder(x)

        x, H, W = pixel_upsample(x, H, W)

        for second_stage_encoder in self.second_stage_encoders:
            x = second_stage_encoder(x)
        
        x, H, W = pixel_upsample(x, H, W)

        for third_stage_encoder in self.third_stage_encoders:
            x = third_stage_encoder(x)
        
        output = self.deconv(x.permute(0, 2, 1).view(-1, self.embed_dim//16, H, W))
        return output
                   

class Discriminator(nn.Module): #Block은 그대로 반영 => Discriminator는 ViT 그대로 사용 
    def __init__(self, args, n_classes: int = 1, d_model: int = 512, d_embedding: int = 256, 
                 n_head: int = 8, dim_feedforward: int = 2048,
                 num_encoder_layer: int = 10, num_decoder_layer: int = 10,
                 img_size: int = 224, patch_size: int = 16, dropout: float = 0.3):
         
        super(Discriminator, self).__init__()
        d_embedding = d_model = args.d_embedding
        img_size = args.img_size

        self.dropout = nn.Dropout(dropout)

        # Image embedding part
        self.patch_embedding = PatchEmbedding(in_channels=3, patch_size=patch_size,
            d_model=d_model, d_embedding=d_embedding, img_size=img_size)

        # Transformer Encoder part
        self_attn = MultiheadAttention(d_model, n_head, dropout=dropout)
        self.encoders = nn.ModuleList([
            TransformerEncoderLayer(d_model, self_attn, dim_feedforward, dropout=dropout) \
                for i in range(num_encoder_layer)])

        # Target linear part (Not averaging)
        self.trg_output_linear = nn.Linear(d_model, d_embedding)
        self.trg_output_norm = nn.LayerNorm(d_embedding, eps=1e-12)
        self.trg_output_linear2 = nn.Linear(d_embedding, n_classes)

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
        
        

class LinearLrDecay(object):
    def __init__(self, optimizer, start_lr, end_lr, decay_start_step, decay_end_step):

        assert start_lr > end_lr
        self.optimizer = optimizer
        self.delta = (start_lr - end_lr) / (decay_end_step - decay_start_step)
        self.decay_start_step = decay_start_step
        self.decay_end_step = decay_end_step
        self.start_lr = start_lr
        self.end_lr = end_lr

    def step(self, current_step):
        if current_step <= self.decay_start_step:
            lr = self.start_lr
        elif current_step >= self.decay_end_step:
            lr = self.end_lr
        else:
            lr = self.start_lr - self.delta * (current_step - self.decay_start_step)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        return lr