import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as f
from .layers import *
from .embed import *

class Pyramid(nn.Module):
    def __init__(self, input_size, seq_size,  ary_size, inner_size, d_model, n_layer, n_head, d_k, d_v, d_inner_hid, agg_type, dropout, \
                 pooling_type='max', granularity=1, local_threshold=0.5, global_threshold=0.5, beta=10, dtw=None):
        super(Pyramid, self).__init__()

        self.input_size = input_size
        self.seq_size = seq_size
        self.d_model = d_model
        self.ary_size = ary_size
        self.num_heads = n_head
        
        self.mask, self.all_size, self.effe_size = get_mask(seq_size, ary_size, inner_size)  # (1,L1...s, T)

        self.depth = len(self.all_size)
        self.indexes = refer_points(self.effe_size, ary_size)

        self.layers = nn.ModuleList([
                    EncoderLayer(d_model, d_inner_hid, n_head, d_k, d_v, dropout, normalize_before=False) for i in range(n_layer)
                    ])

        self.embedding = DataEmbedding(input_size, d_model, seq_size)

        if agg_type == 'conv':
            self.conv_layers = Bottleneck_Construct(d_model, self.depth, ary_size, d_k)
        elif agg_type == 'max':
            self.conv_layers = MaxPooling_Construct(d_model, self.depth, ary_size)
        elif agg_type == 'avg':
            self.conv_layers = AvgPooling_Construct(d_model, self.depth, ary_size)

        self.scorenet = nn.Linear(d_model, 1)
        # self.tenet = nn.Linear(self.all_size[0], d_model) # (T, D)

        # self.temporal_embedding = TemporalEmbedding(seq_size, self.effe_size)


        self.rf_size = self.all_size[0] # the size of receptive field
        self.pooling_type = pooling_type

        self.local_threshold = local_threshold
        self.global_threshold = global_threshold
        self.granularity = granularity
        self.beta = beta
        
        self.dtw = dtw

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                # nn.init.constant_(m.bias, 0) 

        # for m in self.modules():
        #     if isinstance(m, nn.Linear):
        #         init.xavier_normal_(m.weight)
        #         if m.bias is not None:
        #             init.normal_(m.bias)
        #     if isinstance(m, nn.Conv1d):
        #         init.xavier_normal_(m.weight)
        #         if m.bias is not None:
        #             init.normal_(m.bias)
        

    def forward(self, x):   # (B, T, D)
        
        seq_enc = self.embedding(x).transpose(1,2)  # (B, D, T)

        mask = self.mask.repeat(len(seq_enc), self.num_heads, 1, 1).to(x.device) # (B, H, 1, 1)

        padding_size = self.rf_size - seq_enc.shape[2]
        if padding_size > 0:
            seq_enc = f.pad(seq_enc, (0, padding_size), "constant", 0).transpose(1,2)  # (B, T, D)
        
        # Coarse graph construction
        seq_enc = self.conv_layers(seq_enc)  # (B, T+T/C+..., D) 
        # seq_enc += self.tenet(self.all_te.to(x.device))

        # print(seq_enc.shape, mask.shape)
        for i in range(len(self.layers)):
            seq_enc, seq_attn = self.layers[i](seq_enc, mask)  # (B, T+T/C+..., D), (B, N, N)

        # print(self.all_size,self.effe_size)
        out = [seq_enc[:,sum(self.all_size[:i]):sum(self.all_size[:i])+self.effe_size[i]] for i in range(len(self.all_size))]
        out = torch.cat(out, dim=1) # (extract effective samples) # (B, L1+L2+...+Ls, D)

        # te = self.tenet(self.effe_te.to(x.device)) # (1, L1+L2+...+Ls, D)

        return out 


    def get_scores(self, x):
        ret = {}
        out = self.forward(x) # (B, L1+L2+...+Ls, D)


        ret['output'] = out  # (B, L1+L2+...+Ls, D)       

        if self.pooling_type == 'avg':
            _out = torch.mean(out, dim=1) 
        elif self.pooling_type == 'max':
            _out = torch.max(out, dim=1)[0]


        ret['wscore'] = torch.sigmoid(self.scorenet(_out).squeeze(dim=1))
        ret['wpred'] = (ret['wscore'] >= self.global_threshold).type(torch.cuda.FloatTensor)

        # Compute dense scores
        indexes = self.indexes.unsqueeze(0).unsqueeze(-1).repeat(out.size(0), 1, 1, out.size(2)).to(out.device) # (B, T, s, D)
        indexes = indexes.view(out.size(0), -1, out.size(2))  # (B, T*s, D)
        all_enc = torch.gather(out, 1, indexes) # select feature for score (B, T*s)
        all_enc = all_enc.view(out.size(0), self.effe_size[0], -1, out.size(2)) # (B, T, s, D)
        h = torch.mean(all_enc, dim=2) # (B, T, D)
        # h = out[:, :self.seq_size]
        

        d_score = torch.sigmoid(self.scorenet(h)).squeeze(dim=2)  # (B, T)
        ret['dscore'] = d_score
        ret['dpred'] = (ret['dscore'] >= self.local_threshold).type(torch.cuda.FloatTensor)

        return ret

    
    def get_seqlabel(self, actmap, wlabel):   # 若L中存在异常（超过某一阈值），则该段为异常
        actmap *= wlabel.unsqueeze(dim=1).repeat(1, actmap.shape[1]) # filter those normal sample (B, T)
        seqlabel = (actmap >= self.local_threshold).type(torch.cuda.FloatTensor) # (B, T)
        seqlabel = f.pad(seqlabel, (self.granularity - seqlabel.shape[1] % self.granularity, 0), 'constant', 0)
        seqlabel = torch.reshape(seqlabel, (seqlabel.shape[0], -1, int(seqlabel.shape[1] / self.granularity))) # (B, L, T/L)
        seqlabel = torch.max(seqlabel, dim=2)[0] # (B, L)

        seqlabel = torch.cat([torch.zeros(seqlabel.shape[0], 1).cuda(), seqlabel, torch.zeros(seqlabel.shape[0], 1).cuda()], dim=1) # (B, L+2)

        return seqlabel

    def get_alignment(self, label, score):
        # label : batch x pseudo-label length (= B x L)
        # score : batch x time-sereis length (= B x T)
        assert label.shape[0] == score.shape[0]
        assert len(label.shape) == 2  
        assert len(score.shape) == 2

        A = self.dtw.align(label.unsqueeze(dim=1), score.unsqueeze(dim=1))
        indices = torch.max(A, dim=1)[1]
        return torch.gather(label, 1, indices)

    def get_dpred(self, out, wlabel):
        
        indexes = self.indexes.unsqueeze(0).unsqueeze(-1).repeat(out.size(0), 1, 1, out.size(2)).to(out.device) # (B, T, s, D)
        indexes = indexes.view(out.size(0), -1, out.size(2))  # (B, T*s, D)
        all_enc = torch.gather(out, 1, indexes) # select feature for score (B, T*s)
        all_enc = all_enc.view(out.size(0), self.effe_size[0], -1, out.size(2)) # (B, T, s, D)
        enc = torch.mean(all_enc, dim=2) # (B, T, D)
                
        d_h = self.scorenet(enc).squeeze(dim=2) # (B, T)
        dscore = torch.sigmoid(d_h)

        # self.index: (T, s)
        # indexes = self.indexes.unsqueeze(0).repeat(h.size(0), 1, 1).to(h.device) # (B, T, s)
        # indexes = indexes.view(h.size(0), -1)  # (B, T*s, D)
        # all_enc = torch.gather(h, 1, indexes) # select feature for score (B, T*s)
        # all_enc = all_enc.view(h.size(0), self.effe_size[0], -1) # (B, T, s)
        
        # d_h = torch.sum(all_enc, dim=-1) # (B, T)
        # # dscore = torch.sum(all_enc, dim=-1)
        # dscore = torch.sigmoid(d_h)
        # d_h = h[:,:self.seq_size]
        
        with torch.no_grad():
            # Activation map
            actmap = d_h
            actmin = torch.min(actmap, dim=1)[0]
            actmap = actmap - actmin.unsqueeze(dim=1)
            actmax = torch.max(actmap, dim=1)[0]
            actmap = actmap / actmax.unsqueeze(dim=1)
            # Sequential labels
            seqlabel = self.get_seqlabel(actmap, wlabel)

        return self.get_alignment(seqlabel, dscore)

