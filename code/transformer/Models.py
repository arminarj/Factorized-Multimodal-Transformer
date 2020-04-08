import copy

import torch
import torch.nn as nn

from consts import global_consts as gc
from transformer.Layers import EncoderLayer
from transformer.Modules import PositionalEncoder
from transformer.SubLayers import Norm


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def get_pad_mask(seq):
    # [batch, seq_len, dim]
    assert seq.dim() == 3
    non_pad_mask = torch.abs(seq).sum(-1).eq(0)
    return non_pad_mask.unsqueeze(-1)

class TransformerEncoder(nn.Module):
    def __init__(self, d_model, n_layers, dropout):
        super().__init__()
        self.n_layers = n_layers
        dim_l, dim_a, dim_v = d_model
        self.pe_l = PositionalEncoder(dim_l, dropout=dropout)
        self.pe_a = PositionalEncoder(dim_a, dropout=dropout)
        self.pe_v = PositionalEncoder(dim_v, dropout=dropout)

        self.layers = get_clones(EncoderLayer(d_model, dropout), n_layers)
        self.out = nn.Linear(sum(d_model), sum(d_model))
        self.norm = Norm(sum(d_model))


    def forward(self, inputs):
        x_l, x_a, x_v = inputs
        x_list = []
        mask_list = []
        for mod in ['l', 'a', 'v']:
            x_mod = locals()['x_%s' % mod]
            # get position encoded x_mod
            x_list.append(getattr(self, 'pe_%s' % mod)(x_mod))
            mask_mod = get_pad_mask(x_mod)
            locals()['mask_%s' % mod] = mask_mod
            mask_list.append(mask_mod)
        for mod1, mod2 in [('l', 'a'), ('l', 'v'), ('a', 'v')]:
            # only when both masks are 1 at a position, the position will be all-zeros for both dimensions
            mask_list.append(locals()['mask_%s' % mod1] * locals()['mask_%s' % mod2])
        mask_list.append(locals()['mask_l'] * locals()['mask_a'] * locals()['mask_v'])
        x_concat = torch.cat(x_list, -1)
        for i in range(self.n_layers):
            x_concat = self.layers[i](x_concat, mask_list)
        output = self.out(x_concat)
        return self.norm(output)
