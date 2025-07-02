import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from params import args


# ****************************** About plot **************************************


def draw(y, save=True, title=None):
    fig = plt.figure()
    plt.plot(y)
    if title is not None:
        plt.title(title)
    if save:
        fig.savefig('./checkpoint/fig/' + title + '.jpg')
    else:
        plt.show()
    plt.close(fig)


def sub_figure(y, save=True, title=None, label=None):
    """
    >>> sub_figure(y=y, x_ax=x, title='test')
    """
    n, length = y.shape
    fig, ax = plt.subplots()
    for i in range(n):
        plt.subplot(n, 1, i + 1)
        plt.plot(y[i], label=str(i + 1))
    fig.suptitle(title)
    if save:
        fig.savefig('./checkpoint/fig/' + title + '.jpg')
    else:
        plt.show()
    plt.close()


def get_name(args, acc):
    name = './checkpoint/dic/' + args.log_name
    # if args.gcu:
    #     name += '_ffc'
    # if args.att:
    #     name += '_att'

    # name += '_' + str(int(acc * 10000)) + '.pth'
    name += '.pth'

    return name


# ****************************** About Data **********************************
def max_min(x):
    """
    x: torch.FloatTenosr
    """
    max, min = torch.max(x, dim=2)[0], torch.min(x, dim=2)[0]
    max, min = max.unsqueeze(-1), min.unsqueeze(-1)
    norm = (x - min) / (max - min)
    return norm


def wgn(x: torch, snr: float):
    snr = pow(10, snr/10.0)
    xpower = pow(x, 2)
    xpower = torch.sum(xpower, dim=-1) / args.length
    npower = xpower / snr
    return torch.randn(x.shape) * torch.sqrt(npower.unsqueeze(-1))

# ***************************** About Model **********************************

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3., inplace=self.inplace) / 6.


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        out = F.relu6(x + 3., self.inplace) / 6.
        return out * x

class GradBlocker(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha=1):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output * ctx.alpha

        return output, None


def gcu(x):
    c = x.shape[1]
    ffted = torch.fft.fft(x)
    yi, yr = torch.abs(ffted.imag), torch.abs(ffted.real)
    # (batch, c*2, l//2+1)
    # ffted = torch.cat((yi, yr), dim=1)
    # return torch.abs(ffted)
    return yi, yr


if __name__ == '__main__':
    pass
