# Import PyTorch
import torch
from torch import nn
from torch import Tensor
from torch.nn import functional as F
from torch.cuda.amp.autocast_mode import autocast
from torch.nn.modules.activation import MultiheadAttention
# Import custom modules
from ..transformer.embedding import PatchEmbedding, TransformerEmbedding
from ..transformer.layer import TransformerEncoderLayer, TransformerDecoderLayer

class Vision_Transformer(nn.Module):
    def __init__(self, trg_vocab_num: int, d_model: int = 512, d_embedding: int = 256, 
                 n_head: int = 8, dim_feedforward: int = 2048,
                 num_encoder_layer: int = 10, num_decoder_layer: int = 10,
                 img_size: int = 224, patch_size: int = 16, dropout: float = 0.3):
    
        super(Vision_Transformer, self).__init__()

        self.dropout = nn.Dropout(dropout)

        # Image embedding part
        self.patch_embedding = PatchEmbedding(in_channels=3, patch_size=patch_size,
            d_model=d_model, d_embedding=d_embedding, img_size=img_size)

        # Text embedding part
        self.trg_embedding = TransformerEmbedding(trg_vocab_num, d_model, d_embedding, 
            pad_idx=self.pad_idx, max_len=self.src_max_len, dropout=dropout)

        # Transformer Encoder part
        self_attn = MultiheadAttention(d_model, n_head, dropout=dropout)
        self.encoders = nn.ModuleList([
            TransformerEncoderLayer(d_model, self_attn, dim_feedforward, dropout=dropout) \
                for i in range(num_encoder_layer)])

        # Transformer Decoder part
        self_attn = MultiheadAttention(d_model, n_head, dropout=dropout)
        decoder_mask_attn = MultiheadAttention(d_model, n_head, dropout=dropout)
        self.decoders = nn.ModuleList([
            TransformerDecoderLayer(d_model, self_attn, decoder_mask_attn,
                dim_feedforward, dropout=dropout) for i in range(num_decoder_layer)])

        # Target linear part (Not averaging)
        self.trg_output_linear = nn.Linear(d_model, d_embedding)
        self.trg_output_norm = nn.LayerNorm(d_embedding, eps=1e-12)
        self.trg_output_linear2 = nn.Linear(d_embedding, trg_vocab_num)

    @autocast
    def forward(self, src_img: Tensor, trg_text: Tensor, tgt_mask: Tensor = None) -> Tensor:
        # Image embedding
        encoder_out = self.patch_embedding(src_img).transpose(0, 1)

        # Text embedding
        tgt_key_padding_mask = (trg_text == self.pad_idx)
        decoder_out = self.text_embedding(trg_text).transpose(0, 1)
        
        # Transformer Encoder
        for encoder in self.encoders:
            encoder_out = encoder(encoder_out)

        # Transformer Decoder
        for decoder in self.decoders:
            decoder_out = decoder(decoder_out, encoder_out, tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask)

        # Target linear
        decoder_out = decoder_out.transpose(0, 1).contiguous()
        decoder_out = self.trg_output_norm(self.dropout(F.gelu(self.trg_output_linear(decoder_out))))
        decoder_out = self.trg_output_linear2(decoder_out)
        return decoder_out