import math
import torch
from torch.functional import align_tensors
import torch.nn as nn
from torch.nn.modules.linear import Linear
import torch.nn.functional as F



def get_mask(seq_size, ary_size, inner_size):
    """Get the attention mask of PAM"""
    # Get the size of all layers
    all_size = []
    effe_size = []

    s = math.ceil(math.log(seq_size, ary_size))
    all_size.append(int(math.pow(ary_size, s)))
    effe_size.append(seq_size)

    for i in range(s):
        layer_size = all_size[i] // ary_size
        all_size.append(layer_size)  # (输出每层的size)
        effe_size.append(math.ceil(effe_size[i] / ary_size)) # (输出每层有效的size)

    seq_length = sum(all_size)
    mask = torch.zeros(seq_length, seq_length) 


    # get intra-scale mask
    inner_window = inner_size // 2
    for layer_idx in range(len(all_size)):
        start = sum(all_size[:layer_idx]) 
        for i in range(start, start + effe_size[layer_idx]):
            left_side = max(i - inner_window, start)
            right_side = min(i + inner_window + 1, start + effe_size[layer_idx]) # 填充的部分不计算attn
            mask[i, left_side:right_side] = 1

    # get inter-scale mask
    for layer_idx in range(1, len(all_size)):
        start = sum(all_size[:layer_idx])
        for i in range(start, start + effe_size[layer_idx]):
            left_side = (start - all_size[layer_idx - 1]) + (i - start) * ary_size
            # if i == ( start + all_size[layer_idx] - 1):
            #     right_side = start
            # else:
            #     right_side = (start - all_size[layer_idx - 1]) + (i - start + 1) * ary_size
            right_side = min( (start - all_size[layer_idx - 1]) + (i - start + 1) * ary_size, ((start - all_size[layer_idx - 1]) + effe_size[layer_idx - 1]) )  # 填充的部分不计算attn
            mask[i, left_side:right_side] = 1
            mask[left_side:right_side, i] = 1

    mask = (1 - mask).bool()

    # effe_te = []
    # for i in range(len(all_size)):
    #     tmp = torch.zeros((effe_size[i], all_size[0]))
    #     # s = sum(all_size[:i])
    #     step = int(math.pow(ary_size, i))
    #     for j in range(len(tmp)):
    #         start = int(math.pow(ary_size, i)) * j
    #         tmp[j, start:start + step] = 1
    #     effe_te.append(tmp)
    # effe_te = torch.cat(effe_te, dim=0).unsqueeze(0) # (1, L1+L2+...+L3, T)

    # all_te = []
    # for i in range(len(all_size)):
    #     tmp = torch.zeros((all_size[i], all_size[0]))
    #     # s = sum(all_size[:i])
    #     step = int(math.pow(ary_size, i))
    #     for j in range(len(tmp)):
    #         start = int(math.pow(ary_size, i)) * j
    #         tmp[j, start:start + step] = 1
    #     tmp[:,effe_size[0]:]=0 # clear those 
    #     all_te.append(tmp)
    # all_te = torch.cat(all_te, dim=0).unsqueeze(0) # (1, L1+L2+...+L3, T)

    # effe_te = [all_te[:,sum(all_size[:i]):sum(all_size[:i])+effe_size[i]] for i in range(len(all_size))]
    # effe_te = torch.cat(effe_te, dim=1) # (extract effective samples) # (1, L1+L2+...+Ls, D)

    return mask, all_size, effe_size


def refer_points(effe_sizes, ary_size):
    """Gather features from PAM's pyramid sequences"""
    input_size = effe_sizes[0]
    indexes = torch.zeros(input_size, len(effe_sizes))

    for i in range(input_size):
        indexes[i][0] = i
        former_index = i
        for j in range(1, len(effe_sizes)):
            start = sum(effe_sizes[:j])
            inner_layer_idx = former_index - (start - effe_sizes[j - 1])
            former_index = start + min(inner_layer_idx // ary_size, effe_sizes[j] - 1)
            indexes[i][j] = former_index

    # indexes = indexes.unsqueeze(0).unsqueeze(3) # each point有关的segment (1, T, s, 1)

    return indexes.long()

class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, temperature, attn_dropout=0.2):
        super().__init__()

        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    """ Multi-Head Attention module """

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1, normalize_before=True):
        super().__init__()

        self.normalize_before = normalize_before
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        nn.init.xavier_uniform_(self.w_qs.weight)
        nn.init.xavier_uniform_(self.w_ks.weight)
        nn.init.xavier_uniform_(self.w_vs.weight)

        self.fc = nn.Linear(d_v * n_head, d_model)
        nn.init.xavier_uniform_(self.fc.weight)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5, attn_dropout=dropout)

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q
        if self.normalize_before:
            q = self.layer_norm(q)

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            if len(mask.size()) == 3:
                mask = mask.unsqueeze(1)  # For head axis broadcasting.

        output, attn = self.attention(q, k, v, mask=mask)  # ()

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        output = output.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        output = self.dropout(self.fc(output))
        output += residual

        if not self.normalize_before:
            output = self.layer_norm(output)
        return output, attn


class PositionwiseFeedForward(nn.Module):
    """ Two-layer position-wise feed-forward neural network. """

    def __init__(self, d_in, d_hid, dropout=0.1, normalize_before=True):
        super().__init__()

        self.normalize_before = normalize_before

        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)

        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        #self.layer_norm = GraphNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        if self.normalize_before:
            x = self.layer_norm(x)

        x = F.gelu(self.w_1(x))
        x = self.dropout(x)
        x = self.w_2(x)
        x = self.dropout(x)
        x = x + residual

        if not self.normalize_before:
            x = self.layer_norm(x)
        return x


class EncoderLayer(nn.Module):
    """ Compose with two layers """

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1, normalize_before=True,  q_k_mask=None, k_q_mask=None):
        super(EncoderLayer, self).__init__()
        
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout, normalize_before=normalize_before)

        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout, normalize_before=normalize_before)

    def forward(self, enc_input, slf_attn_mask=None):

        enc_output, enc_slf_attn = self.slf_attn(enc_input, enc_input, enc_input, mask=slf_attn_mask)

        enc_output = self.pos_ffn(enc_output)

        return enc_output, enc_slf_attn


class ConvLayer(nn.Module):
    def __init__(self, c_in, window_size):
        super(ConvLayer, self).__init__()
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=window_size,
                                  stride=window_size)
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)


    def forward(self, x):
        x = self.downConv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x

class Bottleneck_Construct(nn.Module):
    """Bottleneck convolution CSCM"""
    def __init__(self, d_model, depth, ary_size,  d_inner):
        super(Bottleneck_Construct, self).__init__()
        
        self.conv_layers = []
        for i in range(depth-1):
            self.conv_layers.append(ConvLayer(d_inner, ary_size))
        self.conv_layers = nn.ModuleList(self.conv_layers)

        self.up = Linear(d_inner, d_model)
        self.down = Linear(d_model, d_inner)
        self.norm = nn.LayerNorm(d_model)
        
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)


    def forward(self, enc_input):

        temp_input = self.down(enc_input).permute(0, 2, 1) # (B, d, T)
        all_inputs = []
        for i in range(len(self.conv_layers)):
            temp_input = self.conv_layers[i](temp_input)  # (B, d, T/C)
            all_inputs.append(temp_input)

        all_inputs = torch.cat(all_inputs, dim=2).transpose(1, 2) 
        all_inputs = self.up(all_inputs) # (B, T/C +T/C2 + T/C3, D)
        all_inputs = torch.cat([enc_input, all_inputs], dim=1)

        all_inputs = self.norm(all_inputs) # (B, T + T/C +T/C2 + T/C3, D)

        return all_inputs


class MaxPooling_Construct(nn.Module):
    """Max pooling CSCM"""
    def __init__(self, d_model, depth, ary_size):
        super(MaxPooling_Construct, self).__init__()

        self.pooling_layers = []
        for i in range(depth-1):
            self.pooling_layers.append(nn.MaxPool1d(kernel_size=ary_size))
        self.pooling_layers = nn.ModuleList(self.pooling_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, enc_input):
        all_inputs = []
        enc_input = enc_input.transpose(1, 2).contiguous()
        all_inputs.append(enc_input)

        for layer in self.pooling_layers:
            enc_input = layer(enc_input)
            all_inputs.append(enc_input)

        all_inputs = torch.cat(all_inputs, dim=2).transpose(1, 2)
        all_inputs = self.norm(all_inputs)

        return all_inputs


class AvgPooling_Construct(nn.Module):
    """Average pooling CSCM"""
    def __init__(self, d_model, depth, ary_size):
        super(AvgPooling_Construct, self).__init__()
        
        self.pooling_layers = []
        for i in range(depth-1):
            self.pooling_layers.append(nn.AvgPool1d(kernel_size=ary_size))
        self.pooling_layers = nn.ModuleList(self.pooling_layers)
        
        self.norm = nn.LayerNorm(d_model)

    def forward(self, enc_input):
        all_inputs = []
        enc_input = enc_input.transpose(1, 2).contiguous()
        all_inputs.append(enc_input)

        for layer in self.pooling_layers:
            enc_input = layer(enc_input)
            all_inputs.append(enc_input)

        all_inputs = torch.cat(all_inputs, dim=2).transpose(1, 2)
        all_inputs = self.norm(all_inputs)

        return all_inputs

