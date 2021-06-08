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
                 pad_id: int = 0, num_encoder_layer: int = 10, num_decoder_layer: int = 10,
                 dropout: float = 0.3, embedding_dropout: float = 0.15, 
                 triple_patch: bool = False, parallel: bool = False):
    
        super(Vision_Transformer, self).__init__()

        # Hyper-parameter setting
        self.pad_id = pad_id

        # Parallel Transformer
        self.parallel = parallel
        if self.parallel:
            self.num_common_layers = min(num_encoder_layer, num_decoder_layer)
            self.num_encoder_nonparallel = num_encoder_layer - self.num_common_layers

        # Image embedding part
        self.patch_embedding = PatchEmbedding(in_channels=3, patch_size=patch_size,
            d_model=d_model, img_size=img_size, triple_patch=triple_patch)

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

        # Target linear part
        self.trg_dropout = nn.Dropout(dropout)
        self.trg_output_linear = nn.Linear(d_model, d_embedding)
        self.trg_output_norm = nn.LayerNorm(d_embedding, eps=1e-12)
        self.trg_output_linear2 = nn.Linear(d_embedding, trg_vocab_num)

        # Initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.kaiming_uniform_(p) 

    @autocast()
    def forward(self, src_img: Tensor, trg_text: Tensor, tgt_mask: Tensor, 
                non_pad_position: Tensor = None) -> Tensor:
        # Image embedding
        encoder_out = self.patch_embedding(src_img).transpose(0, 1)

        # Text embedding
        tgt_key_padding_mask = (trg_text == self.pad_id)
        decoder_out = self.text_embedding(trg_text).transpose(0, 1)

        # Parallel mode
        if self.parallel:
            # Transformer Encoder
            for encoder in self.encoders[:self.num_encoder_nonparallel+1]:
                encoder_out = encoder(encoder_out)

            # Parallel Transformer
            for encoder, decoder in zip(
                    self.encoders[self.num_encoder_nonparallel+1:],
                    self.decoders[:self.num_common_layers-1]):
                decoder_out = decoder(decoder_out, encoder_out, tgt_mask=tgt_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask)
                encoder_out = encoder(encoder_out)

            # Transformer Decoder
            for decoder in self.decoders[self.num_common_layers-1:]:
                decoder_out = decoder(decoder_out, encoder_out, tgt_mask=tgt_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask)

        # Non-Parallel (Original) mode
        else:
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
        decoder_out = self.trg_output_norm(self.trg_dropout(F.gelu(self.trg_output_linear(decoder_out))))
        decoder_out = self.trg_output_linear2(decoder_out)
        return decoder_out

    @staticmethod
    def generate_square_subsequent_mask(sz, device):
        mask = torch.tril(torch.ones(sz, sz, dtype=torch.float, device=device))
        mask = mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, 0.0)
        return mask