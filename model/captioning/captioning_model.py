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
                 img_size: int = 224, patch_size: int = 16, max_len: int = 300,
                 pad_id: int = 0, unk_id: int = 3, bos_id: int = 1, eos_id: int = 2,
                 num_encoder_layer: int = 10, num_decoder_layer: int = 10,
                 dropout: float = 0.3, embedding_dropout: float = 0.15):
    
        super(Vision_Transformer, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.pad_id = pad_id

        # Image embedding part
        self.patch_embedding = PatchEmbedding(in_channels=3, patch_size=patch_size,
            d_model=d_model, d_embedding=d_embedding, img_size=img_size)

        # Text embedding part
        self.text_embedding = TransformerEmbedding(trg_vocab_num, d_model, d_embedding, 
            pad_idx=self.pad_id, max_len=max_len, embedding_dropout=embedding_dropout)

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

    @autocast()
    def forward(self, src_img: Tensor, trg_text: Tensor, tgt_mask: Tensor, 
                non_pad_position: Tensor = None) -> Tensor:
        # Image embedding
        encoder_out = self.patch_embedding(src_img).transpose(0, 1)

        # Text embedding
        tgt_key_padding_mask = (trg_text == self.pad_id)
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
        if non_pad_position is not None:
            decoder_out = decoder_out[non_pad_position]
        decoder_out = self.trg_output_norm(self.dropout(F.gelu(self.trg_output_linear(decoder_out))))
        decoder_out = self.trg_output_linear2(decoder_out)
        return decoder_out

    @staticmethod
    def generate_square_subsequent_mask(sz, device):
        mask = torch.tril(torch.ones(sz, sz, dtype=torch.float, device=device))
        mask = mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, 0.0)
        return mask