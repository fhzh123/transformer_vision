# Import PyTorch
import torch
from torch import nn
from torch import Tensor
from torch.nn import functional as F
from torch.nn.modules.activation import MultiheadAttention
# Import custom modules
from .layers import PatchEmbedding, TransformerEncoderLayer
# from .layers import PatchEmbedding
# from .transformer import TransformerEncoder, ClassificationHead

class Trans_GAN(nn.Module):
    def __init__(self, n_classes, d_model=512, d_embedding=256, n_head=8, dim_feedforward=2048,
            num_encoder_layer=10, num_decoder_layer=10, img_size=224, patch_size=16,
            dropout=0.3):
    
        super(Trans_GAN, self).__init__()

        self.dropout = nn.Dropout(dropout)

        # Image embedding part
        self.patch_embedding = PatchEmbedding(in_channels=3, patch_size=patch_size,
            d_model=d_model, d_embedding=d_embedding, img_size=img_size)

        # Transformer Encoder part
        self_attn = MultiheadAttention(d_model, n_head, dropout=dropout)
        self.encoders = nn.ModuleList([
            TransformerEncoderLayer(d_model, self_attn, dim_feedforward, dropout=dropout) \
                for i in range(num_encoder_layer)])

    def forward(self, src_img: Tensor) -> Tensor:
        # Image embedding
        emb_out = self.patch_embedding(src_img).transpose(0, 1)
        
        # Transformer Encoder
        for encoder in self.encoders:
            encoder_out = encoder(emb_out)

        # Target linear
        encoder_out = encoder_out.transpose(0, 1)
        encoder_out = self.trg_output_norm(self.dropout(F.gelu(self.trg_output_linear(encoder_out))))
        encoder_out = self.trg_output_linear2(encoder_out)
        return encoder_out