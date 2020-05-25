import torch.nn as nn

from consts import global_consts as gc
from transformer import Models


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        conf = gc.config
        proj_dim_a = conf['proj_dim_a']
        proj_dim_v = conf['proj_dim_v']
        self.proj_a = nn.Linear(gc.dim_a, proj_dim_a)
        self.proj_v = nn.Linear(gc.dim_v, proj_dim_v)
        self.transformer_encoder = Models.TransformerEncoder((gc.dim_l, proj_dim_a, proj_dim_v),
                                                             conf['n_layers'], conf['dropout'])
        dim_total_proj = conf['dim_total_proj']
        dim_total = gc.dim_l + proj_dim_a + proj_dim_v
        self.gru = nn.GRU(input_size=dim_total, hidden_size=dim_total_proj)
        if gc.dataset == 'iemocap':
            final_out_dim = 2 * len(gc.best.emos)
        elif gc.dataset == 'pom':
            final_out_dim = len(gc.best.pom_cls)
        elif gc.dataset == 'pom':
            final_out_dim = 6
        else:
            final_out_dim = 1
        self.finalW = nn.Linear(dim_total_proj, final_out_dim)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, words, covarep, facet, inputLens):
        state = self.transformer_encoder((words, self.proj_a(covarep), self.proj_v(facet)))
        # convert input to GRU from shape [batch_size, seq_len, input_size] to [seq_len, batch_size, input_size]
        _, gru_last_h = self.gru(state.transpose(0, 1))
        return self.finalW(gru_last_h.squeeze()).squeeze()
