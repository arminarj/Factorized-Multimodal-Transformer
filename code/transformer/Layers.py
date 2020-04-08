import torch
import torch.nn as nn

from consts import global_consts as gc


from transformer.SubLayers import FeedForward, MultiHeadAttention, Norm


def add_attr(obj, mod, dim, dropout):
    # setattr(obj, 'norm1_%s' % mod, Norm(dim))
    # setattr(obj, 'norm2_%s' % mod, Norm(dim))
    n_heads = gc.config['n_head']
    setattr(obj, 'attn_%s' % mod, MultiHeadAttention(n_heads, dim, dropout=dropout))
    if mod in ['l', 'a', 'v']:
        for head in range(n_heads):
            conv_in_d = 4
            for i, conv_d in enumerate(gc.config['conv_dims']):
                setattr(obj, 'conv_%s_head_%d_%d' % (mod, head, i), nn.Linear(conv_in_d, conv_d))
                conv_in_d = conv_d
            setattr(obj, 'conv_%s_out_%d' % (mod, head), nn.Linear(conv_in_d, 1))


class EncoderLayer(nn.Module):
    def __init__(self, d_model, dropout):
        super().__init__()
        self.dim_l, self.dim_a, self.dim_v = d_model
        for mod in ['l', 'a', 'v']:
            dim = getattr(self, 'dim_%s' % mod)
            add_attr(self, mod, dim, dropout)
        for mod1, mod2 in [('l', 'a'), ('l', 'v'), ('a', 'v')]:
            dim = getattr(self, 'dim_%s' % mod1) + getattr(self, 'dim_%s' % mod2)
            setattr(self, 'dim_%s_%s' % (mod1, mod2), dim)
            add_attr(self, '%s_%s' % (mod1, mod2), dim, dropout)

        self.dim_l_a_v = self.dim_l + self.dim_a + self.dim_v
        add_attr(self, 'l_a_v', self.dim_l_a_v, dropout)
        self.dropout = nn.Dropout(dropout)
        total_dim = sum(d_model)
        self.ff = FeedForward(total_dim, d_ff=gc.config['ff_dim_final'], dropout=dropout)
        self.norm1 = Norm(total_dim)
        self.norm2 = Norm(total_dim)

    def forward(self, x_concat, mask_list):
        mask_l, mask_a, mask_v, mask_l_a, mask_l_v, mask_a_v, mask_l_a_v = mask_list
        x_concat_normed = self.norm1(x_concat)

        cum_dim = 0
        for mod in ['l', 'a', 'v']:
            dim_m = getattr(self, 'dim_%s' % mod)
            locals()['x_%s' % mod] = x_concat[:, :, cum_dim:cum_dim + dim_m]
            locals()['x_%s_normed' % mod] = x_concat_normed[:, :, cum_dim:cum_dim + dim_m]
            for head in range(gc.config['n_head']):
                locals()['x_%s_list_%d' % (mod, head)] = []
            cum_dim += dim_m

        for mod in ['l', 'a', 'v', 'l_a', 'l_v', 'a_v', 'l_a_v']:
            mods_list = mod.split('_')
            # if len(mods_list) == 1:
            #     x = locals()['x_%s' % mod]
            #     x2 = locals()['x_%s_normed' % mod]
            # else:
            x_pre_cat = []
            for m in mods_list:
                x_pre_cat.append(locals()['x_%s' % m])
            x = torch.cat(x_pre_cat, -1)
            # x2 = getattr(self, 'norm1_%s' % mod)(x)
            x_norm_pre_cat = []
            for m in mods_list:
                x_norm_pre_cat.append(locals()['x_%s_normed' % m])
            x2 = torch.cat(x_norm_pre_cat, -1)

            mask = locals()['mask_%s' % mod]
            multihead_attns = getattr(self, 'attn_%s' % mod)(x2, x2, x2, mask)
            for head in range(gc.config['n_head']):
                x = x + self.dropout(multihead_attns[head])
                # x2 = getattr(self, 'norm2_%s' % mod)(x)
                cum_dim = 0
                for m in mods_list:
                    dim_m = getattr(self, 'dim_%s' % m)
                    x_out_m = x[:, :, cum_dim:cum_dim+dim_m]
                    (locals()['x_%s_list_%d' % (m, head)]).append(x_out_m)
                    cum_dim += dim_m
        x_list = []
        for mod in ['l', 'a', 'v']:
            x_mod_list = []
            for head in range(gc.config['n_head']):
                x_cat_mod = torch.stack(locals()['x_%s_list_%d' % (mod, head)], dim=3)
                for i in range(len(gc.config['conv_dims'])):
                    x_cat_mod = getattr(self, 'conv_%s_head_%d_%d' % (mod, head, i))(x_cat_mod)
                x_cat_mod = getattr(self, 'conv_%s_out_%d' % (mod, head))(x_cat_mod).squeeze()
                x_mod_list.append(x_cat_mod)
            x_list.append(torch.sum(torch.stack(x_mod_list, -1), -1))
        x2 = self.norm2(torch.cat(x_list, -1))
        return x_concat + self.dropout(self.ff(x2))
