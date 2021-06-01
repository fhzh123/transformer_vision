import math
import torch
from torch import nn

from ..util.misc import NestedTensor


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, 
                num_pos_feats=64, 
                temperature=10000, 
                normalize=False, 
                scale=None):
        super().__init__()

        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize

        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")

        if scale is None:
            scale = 2 * math.pi

        self.scale = scale

    def forward(self, tensor_list: NestedTensor):

        """
        Output example)

        Position Embedding shape : torch.Size([2, 256, 36, 24])

        tensor([[[ True,  True,  True,  ..., False, False, False],
         [ True,  True,  True,  ..., False, False, False],
         [ True,  True,  True,  ..., False, False, False],
         ...,
         [ True,  True,  True,  ..., False, False, False],
         [ True,  True,  True,  ..., False, False, False],
         [ True,  True,  True,  ..., False, False, False]],

        [[ True,  True,  True,  ...,  True,  True,  True],
         [ True,  True,  True,  ...,  True,  True,  True],
         [ True,  True,  True,  ...,  True,  True,  True],
         ...,
         [False, False, False,  ..., False, False, False],
         [False, False, False,  ..., False, False, False],
         [False, False, False,  ..., False, False, False]]], device='cuda:0')
y_embed : tensor([[[ 1.,  1.,  1.,  ...,  0.,  0.,  0.],
         [ 2.,  2.,  2.,  ...,  0.,  0.,  0.],
         [ 3.,  3.,  3.,  ...,  0.,  0.,  0.],
         ...,
         [23., 23., 23.,  ...,  0.,  0.,  0.],
         [24., 24., 24.,  ...,  0.,  0.,  0.],
         [25., 25., 25.,  ...,  0.,  0.,  0.]],

        [[ 1.,  1.,  1.,  ...,  1.,  1.,  1.],
         [ 2.,  2.,  2.,  ...,  2.,  2.,  2.],
         [ 3.,  3.,  3.,  ...,  3.,  3.,  3.],
         ...,
         [21., 21., 21.,  ..., 21., 21., 21.],
         [21., 21., 21.,  ..., 21., 21., 21.],
         [21., 21., 21.,  ..., 21., 21., 21.]]], device='cuda:0')
x_embed : tensor([[[ 1.,  2.,  3.,  ..., 29., 29., 29.],
         [ 1.,  2.,  3.,  ..., 29., 29., 29.],
         [ 1.,  2.,  3.,  ..., 29., 29., 29.],
         ...,
         [ 1.,  2.,  3.,  ..., 29., 29., 29.],
         [ 1.,  2.,  3.,  ..., 29., 29., 29.],
         [ 1.,  2.,  3.,  ..., 29., 29., 29.]],

        [[ 1.,  2.,  3.,  ..., 31., 32., 33.],
         [ 1.,  2.,  3.,  ..., 31., 32., 33.],
         [ 1.,  2.,  3.,  ..., 31., 32., 33.],
         ...,
         [ 0.,  0.,  0.,  ...,  0.,  0.,  0.],
         [ 0.,  0.,  0.,  ...,  0.,  0.,  0.],
         [ 0.,  0.,  0.,  ...,  0.,  0.,  0.]]], device='cuda:0')
        """

        # Position Embeding input tensor shape : torch.Size([2, 2048, 36, 24])
        # Position Embeding mask shape : torch.Size([2, 36, 24])
        x = tensor_list.tensors
        mask = tensor_list.mask
        assert mask is not None
        not_mask = ~mask

        # y_embed : 행 방향으로 True인 값을 누적해서 숫자 매김 
        # x_emded : 열 방향으로 True인 값을 누적해서 숫자 매김 
        # 즉, 이미지가 존재하는 부분에 대해서만 숫자 매김, 
        # False, 즉 pad된 부분은 0 
        # y_embed shape : torch.Size([2, 36, 24])
        # x_embed shape : torch.Size([2, 36, 24])
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        # dim_t shape : torch.Size([128])
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        # pos_x shape : torch.Size([2, 36, 24, 128])
        # pos_y shape : torch.Size([2, 36, 24, 128])
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t

        # stacked pos_x shape : torch.Size([2, 36, 24, 128])
        # staked pos_y shape : torch.Size([2, 36, 24, 128])
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)

        # dim=3 기준으로 concat
        # Position Embedding shape : torch.Size([2, 256, 36, 24])
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)

        return pos


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, num_pos_feats=256):
        super().__init__()

        self.row_embed = nn.Embedding(50, num_pos_feats)
        self.col_embed = nn.Embedding(50, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):

        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, tensor_list: NestedTensor):

        x = tensor_list.tensors
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)

        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)

        return pos


def build_position_encoding(args):

    N_steps = args.hidden_dim // 2

    if args.position_embedding in ('v2', 'sine'):
        # TODO find a better way of exposing other arguments
        position_embedding = PositionEmbeddingSine(N_steps, normalize=True)

    elif args.position_embedding in ('v3', 'learned'):
        position_embedding = PositionEmbeddingLearned(N_steps)

    else:
        raise ValueError(f"not supported {args.position_embedding}")

    return position_embedding