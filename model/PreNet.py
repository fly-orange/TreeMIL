import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as f
from .layers import *
from .embed import *

class PyraAE(nn.Module):
    def __init__(self, input_size, seq_size,  ary_size, inner_size, d_model, n_layer, n_head, d_k, d_v, d_inner_hid, dropout):
        super(PyraAE, self).__init__()

        self.input_size = input_size
        self.seq_size = seq_size
        self.d_model = d_model
        self.ary_size = ary_size
        self.num_heads = n_head
        
        self.mask, self.all_size, self.effe_size = get_mask(seq_size, ary_size, inner_size)
        self.depth = len(self.all_size)
        self.indexes = refer_points(self.effe_size, ary_size)

        self.layers = nn.ModuleList([
                    EncoderLayer(d_model, d_inner_hid, n_head, d_k, d_v, dropout, normalize_before=False) for i in range(n_layer)
                    ])

        self.embedding = DataEmbedding(input_size, d_model, seq_size)

        self.conv_layers = Bottleneck_Construct(d_model, self.depth, ary_size, d_k)

        self.recnet = nn.Linear(d_model, input_size, bias=True)


        self.rf_size = self.all_size[0] # the size of receptive field
        

    def forward(self, x):   # (B, T, D)
        
        seq_enc = self.embedding(x).transpose(1,2)  # (B, D, T)

        mask = self.mask.repeat(len(seq_enc), self.num_heads, 1, 1).to(x.device) # (B, H, 1, 1)

        padding_size = self.rf_size - seq_enc.shape[2]
        if padding_size > 0:
            seq_enc = f.pad(seq_enc, (padding_size, 0), "constant", 0).transpose(1,2)  # (B, T, D)
        
        # Coarse graph construction
        seq_enc = self.conv_layers(seq_enc)  # (B, T+T/C+..., D) 
        # print(seq_enc.shape, mask.shape)
        for i in range(len(self.layers)):
            seq_enc, _ = self.layers[i](seq_enc, mask)  # (B, T+T/C+..., D)

        seq_enc = seq_enc[:, :self.seq_size]
        
        seq_rec = self.recnet(seq_enc) # (B, T, D)


        return seq_rec

