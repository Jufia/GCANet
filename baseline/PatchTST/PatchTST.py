'''
code refer to thuml/Time-Series-Library
paper: A Time Series is Worth 64 Words: Long-term Forecasting with Transformers
'''

import torch
from torch import nn
from .utilise import Encoder, EncoderLayer, FullAttention, AttentionLayer, PatchEmbedding

from params import args


class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False): 
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)


class patchtstModel(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2211.14730.pdf
    """

    def __init__(self, patch_len=16, stride=8):
        """
        patch_len: int, patch len for patch_embedding
        stride: int, stride for patch_embedding
        """
        super().__init__()
        self.task_name = 'classification'
        self.seq_len = args.length
        padding = stride
        dropout = 0.2
        d_model = 128

        # patching and embedding
        self.patch_embedding = PatchEmbedding(
            d_model, patch_len, stride, padding, dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, factor=5, attention_dropout=dropout,
                                      output_attention=False), d_model, 16),
                    d_model,
                    256,
                    dropout=dropout,
                    activation='gelu'
                ) for l in range(3)
            ],
            norm_layer=nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        )

        # Prediction Head
        self.head_nf = d_model * int((self.seq_len - patch_len) / stride + 2)
        self.flatten = nn.Flatten(start_dim=-2)
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(self.head_nf * 6, args.class_num)

    def classification(self, x_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)
        # u: [bs * nvars x patch_num x d_model]
        enc_out, n_vars = self.patch_embedding(x_enc)

        # Encoder
        # z: [bs * nvars x patch_num x d_model]
        enc_out, attns = self.encoder(enc_out)
        # z: [bs x nvars x patch_num x d_model]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        # Decoder
        output = self.flatten(enc_out)
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc, mask=None):   
        x_enc = x_enc.permute(0, 2, 1)  # (Batch, Length, Channel)
        dec_out = self.classification(x_enc)
        return dec_out  # [B, N]
    
if __name__ == '__main__':
    x = torch.rand(3, 5, 1024)
    model = patchtstModel()
    y = model(x)
    print(y.shape)
    