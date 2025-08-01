import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from .utilise import DataEmbedding, Inception_Block_V1

from params import args

def FFT_for_Period(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class TimesBlock(nn.Module):
    def __init__(self, d_model):
        super(TimesBlock, self).__init__()
        self.seq_len = args.length
        self.pred_len = 0 # configed in class Exp_Classification
        self.k = 1 # configs.top_k 1 or 2 or 3
        self.d_ff = 64 # configs.d_ff 64 or 32
        num_kernels = 6 # configs.num_kernels, default=6
        # parameter-efficient design
        self.conv = nn.Sequential(
            Inception_Block_V1(d_model, self.d_ff, num_kernels=num_kernels),
            nn.GELU(),
            Inception_Block_V1(self.d_ff, d_model, num_kernels=num_kernels)
        )

    def forward(self, x):
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]
            # padding
            if (self.seq_len + self.pred_len) % period != 0:
                length = (
                                 ((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = (self.seq_len + self.pred_len)
                out = x
            # reshape
            out = out.reshape(B, length // period, period,
                              N).permute(0, 3, 1, 2).contiguous()
            # 2D conv: from 1d Variation to 2d Variation
            out = self.conv(out)
            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.seq_len + self.pred_len), :])
        res = torch.stack(res, dim=-1)
        # adaptive aggregation
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(
            1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        # residual connection
        res = res + x
        return res


class Model(nn.Module):
    """
    Paper link: https://openreview.net/pdf?id=ju_Uqw384Oq
    """

    def __init__(self):
        super(Model, self).__init__()
        self.task_name = 'classification'
        self.seq_len = args.length
        self.layer = 2  # configs.e_layers 2 or 3 or 6
        self.d_model = 32 # configs.d_model 32 or 16
        self.num_class = args.class_num
        enc_in = args.in_channel
        self.embed = 'timeF'
        self.freq = 'h'
        dropout = 0.01
        self.model = nn.ModuleList([TimesBlock(self.d_model)
                                    for _ in range(self.layer)])
        self.enc_embedding = DataEmbedding(enc_in, self.d_model, self.embed, self.freq,
                                           dropout)
        
        self.layer_norm = nn.LayerNorm(self.d_model)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(dropout)
            self.projection = nn.Linear(
                self.d_model * self.seq_len, self.num_class)


    def classification(self, x_enc, x_mark_enc=None):
        # embedding
        enc_out = self.enc_embedding(x_enc, None)  # [B,T,C]
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))

        # Output
        # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.act(enc_out)
        output = self.dropout(output)
        # zero-out padding embeddings
        # output = output * x_mark_enc.unsqueeze(-1)
        # (batch_size, seq_length * d_model)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        x_enc = x_enc.permute(0, 2, 1)
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None
    
if __name__ == '__main__':
    x = torch.rand(13, args.in_channel, args.length)
    model = Model()
    y = model(x)
    print(y.shape)