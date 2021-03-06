from einops.layers.torch import Rearrange
# Import PyTorch
from torch import nn
from torch import Tensor
from torch.nn import functional as F
from torch.cuda.amp.autocast_mode import autocast
from torch.nn.modules.activation import MultiheadAttention
# Import custom modules
from ..transformer.embedding import PatchEmbedding
from ..transformer.layer import TransformerEncoderLayer

class Vision_Transformer(nn.Module):
    def __init__(self, n_classes: int, d_model: int = 512, d_embedding: int = 256, 
                 n_head: int = 8, dim_feedforward: int = 2048,
                 num_encoder_layer: int = 10, img_size: int = 224, patch_size: int = 16, 
                 dropout: float = 0.3, triple_patch: bool = False):
    
        super(Vision_Transformer, self).__init__()

        self.dropout = nn.Dropout(dropout)

        # Image embedding part
        self.patch_embedding = PatchEmbedding(in_channels=3, patch_size=patch_size,
            d_model=d_model, img_size=img_size, triple_patch=triple_patch)

        # Transformer Encoder part
        self_attn = MultiheadAttention(d_model, n_head, dropout=dropout)
        self.encoders = nn.ModuleList([
            TransformerEncoderLayer(d_model, self_attn, dim_feedforward, dropout=dropout) \
                for i in range(num_encoder_layer)])

        # Image upsample part
        self.upsample = nn.Sequential(
            Rearrange('b (h w) e -> b e (h) (w)', 
                h=img_size//patch_size, w=img_size//patch_size),
            nn.ConvTranspose2d(d_model, 3, kernel_size=patch_size, stride=patch_size)
        )

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

        # Image upsample
        # Target linear
        encoder_out = encoder_out.transpose(0, 1)
        cls_token = encoder_out[0]
        cls_token = self.trg_output_norm(self.dropout(F.gelu(self.trg_output_linear(cls_token))))
        cls_token = self.trg_output_linear2(cls_token)
        
        encoder_upsample = encoder_out[1:]
        encoder_upsample = self.upsample(encoder_upsample)

        return cls_token, encoder_upsample