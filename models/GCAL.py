import torch
import torch.nn as nn
import torch.nn.functional as F

from params import args
from utilise import *

alpha = 1


def _weights_init(m):
    if isinstance(m, nn.Conv1d):
        torch.nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm1d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        torch.nn.init.kaiming_uniform_(m.weight)

class channel_fuse(nn.Module):
    def __init__(self, in_chan, num_feature):
        super(channel_fuse, self).__init__()
        self.fuse = nn.Sequential(
            nn.Conv1d(in_channels=in_chan, out_channels=num_feature, kernel_size=1),
            nn.BatchNorm1d(num_feature),
            h_swish(inplace=True),
        )

    def forward(self, x):
        return self.fuse(x)


class GCU(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.AvgPool1d(kernel_size=3, stride=1),
            nn.Conv1d(2*channels, 2*channels, kernel_size=3, groups=2, stride=2),
            nn.BatchNorm1d(2*channels)
        )
        self.conv2 = nn.Sequential(
            nn.AvgPool1d(kernel_size=3, stride=1),
            nn.Conv1d(2*channels, 2*channels, kernel_size=3, groups=channels, stride=2),
            nn.BatchNorm1d(2*channels)
        )
        
    def forward(self, x):
        b, c, l = x.size()
        yi, yr = gcu(x)  # (b, c, l) * 2
        y1 = torch.cat((yi, yr), dim=1) # (b, 2*c, l)
        y2 = torch.cat((yi, yr), dim=-1).reshape(b, c*2, -1)
        y1 = self.conv1(y1)
        y2 = self.conv2(y2)

        return torch.cat((y1, y2), dim=1) # (b, 4*c, l/2)


class agca(nn.Module):
    def __init__(self, exp_size, d=16, divide=4, head=1):
        super(agca, self).__init__()
        self.len = 64
        self.d = d
        self.GCU = GCU(channels=exp_size)

        gcu_cov = nn.Sequential(
            nn.AdaptiveAvgPool1d(d),
            nn.Linear(d, 1),
            nn.Dropout(0.4),
            nn.Conv1d(in_channels=4*exp_size, out_channels=exp_size*d, kernel_size=1, groups=4),
            # nn.Dropout(0.1),
            nn.ReLU(),
        )
                
        self.se = nn.Sequential(
            nn.Linear(1*head*d*exp_size, exp_size//divide),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(exp_size//divide, exp_size),
            h_sigmoid()
        )
        
        self.gca_cov1 = nn.ModuleList(
            [gcu_cov for _ in range(head)]
        ) # (b, 2*c, d)

    def forward(self, x):
        batch, channels, length = x.size()
        p = x.clone()
        if args.blocker:
            beta = 0 if alpha<0.3 else 1
            p = GradBlocker.apply(p, beta)
        y = self.GCU(p) # (b, 4*c, l/2)
        x1 = torch.empty(batch, self.d*channels, 0).to(args.GPU)
        for m in self.gca_cov1:
            xm = m(y) # (b, 2*c, d)
            x1 = torch.cat((x1, xm), dim=-1) # (b, 2*c, head*d)

        y = x1
        out = y.flatten(start_dim=1)
        out = self.se(out)
        out = out.view(batch, channels, 1)

        q = out * x
        q = GradBlocker.apply(q, alpha)
        return q
    

class SeModule(nn.Module):
    def __init__(self, in_size, d=4):
        super(SeModule, self).__init__()
        expand_size =  in_size*d
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(start_dim=1),
            nn.Linear(in_size, in_size, bias=False),
            h_sigmoid()
        )

    def forward(self, x):
        b, c, l = x.shape
        p = x.clone()
        if args.blocker:
            p = GradBlocker.apply(p, 0)
        return x * self.se(p).view(b, c, 1)

# class SeModule(nn.Module):
#     def __init__(self, in_size, d=16):
#         super(SeModule, self).__init__()
#         expand_size =  in_size*d
#         self.se = nn.Sequential(
#             nn.Conv1d(in_channels=in_size, out_channels=4*in_size, kernel_size=3),
#             nn.AdaptiveAvgPool1d(d),
#             nn.Linear(d, 1),
#             nn.Conv1d(4*in_size, in_size*d, kernel_size=1),
#             nn.ReLU(),
#             nn.Flatten(start_dim=1),
#             nn.Linear(d*in_size, in_size),
#             h_sigmoid()
#         )

#     def forward(self, x):
#         b, c, l = x.shape
#         p = x.clone()
#         if args.blocker:
#             p = GradBlocker.apply(p, 0)
#         return x * self.se(p).view(b, c, 1)


class Channel_attention(nn.Module):
    def __init__(self, chann, if_gcu, head):
        super(Channel_attention, self).__init__()
        if if_gcu:
            self.ca = agca(chann, head=head)
        else:
            self.ca = SeModule(chann, d=head)

    def forward(self, x):
        return self.ca(x)


class Global_Convolution(nn.Module):
    def __init__(self, in_channels, if_gcu, if_att=True, len=args.length):
        super(Global_Convolution, self).__init__()
        self.gcu = if_gcu
        self.att = if_att
        if if_gcu:
            self.gcu = GCU(channels=in_channels) # 2 * (b, 2*c, l/2)

        else:
            self.front = nn.Sequential(
                nn.Conv1d(in_channels=in_channels, out_channels=4*in_channels,
                        kernel_size=3, stride=2),
                nn.BatchNorm1d(4*in_channels),
                nn.ReLU(inplace=True),
            )

        self.conv = nn.Sequential(
            nn.AdaptiveMaxPool1d(len),
            nn.Linear(len, len//8),
            nn.ReLU(),
            nn.Linear(len//8, len//2),
            nn.BatchNorm1d(4*in_channels),
            h_sigmoid(),
            channel_fuse(in_chan=4*in_channels, num_feature=in_channels),
        )

        if self.att:
            self.atten = agca(in_channels, gcu=if_gcu, head=args.head)

    def forward(self, x):
        batch, c, l = x.size()
        if self.gcu:
            y = self.gcu(x) # (b, c*4, l/2)
        
        else:
            y = self.front(x)

        y = self.conv(y)  # (batch, c, l)
        
        if self.att:
            y = self.atten(y)
        
        return y # (b, c, l)


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernal_size, stride, nonLinear, att, ifgca, head, exp_size):
        super(Block, self).__init__()
        self.out_channels = out_channels
        self.nonLinear = nonLinear
        self.att = att
        padding = (kernal_size - 1) // 2

        self.use_connect = stride == 1 and in_channels == out_channels

        if self.nonLinear == "RE":
            activation = nn.ReLU
        else:
            activation = h_swish

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, exp_size, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(exp_size),
            activation(inplace=True)
        )
        self.depth_conv = nn.Sequential(
            nn.Conv1d(exp_size, exp_size, kernel_size=kernal_size, stride=stride, padding=padding, groups=exp_size),
            nn.Dropout(0.2),
            nn.BatchNorm1d(exp_size),
        )

        if self.att:
            self.ca = Channel_attention(exp_size, if_gcu=ifgca, head=head)
            # if ifgca:
            #     self.agca = agca(exp_size, gcu=True, head=head)
            # else:
            #     self.agca = SeModule(exp_size)

        self.point_conv = nn.Sequential(
            nn.Conv1d(exp_size, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(out_channels),
            activation(inplace=True)
        )

    def forward(self, x):
        out = self.conv(x)
        out = self.depth_conv(out)

        if self.att:
            out = self.ca(out)

        out = self.point_conv(out)

        if self.use_connect:
            return x + out
        else:
            return out


class GCALNet(nn.Module):
    def __init__(self, new_class=args.class_num, in_chan=args.in_channel, dropout_rate=0.3):
        super(GCALNet, self).__init__()
        self.num_classes = new_class

        layers = args.layer

        self.ffc = Global_Convolution(in_chan, if_gcu=args.gcug, if_att=args.attg, len=args.length//2)
        self.fuse = channel_fuse(in_chan=in_chan, num_feature=10)

        self.block = nn.Sequential(
            Block(10, 24, 15, 2, 'RE', args.attb, args.gcub, args.head, 24)
        )

        out_conv2_in = 24
        out_conv2_out = 18
        self.out_conv2 = nn.Sequential(
            nn.Conv1d(out_conv2_in, out_conv2_out, kernel_size=1, stride=1),
            h_swish(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Conv1d(out_conv2_out, self.num_classes, kernel_size=1, stride=1),
        )

        self.apply(_weights_init)

    def forward(self, x, a=1):
        alpha = a
        x = self.ffc(x)
        out = self.fuse(x)
        out = self.block(out)
        batch, channels, length = out.size()
        out = F.adaptive_avg_pool1d(out, output_size=1)
        out = self.out_conv2(out).view(batch, -1)
        return out


if __name__ == '__main__':
    x = torch.rand(12, 6, 1024)
    m = GCALNet()
    y = m(x)
    print(y.shape)