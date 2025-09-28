import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from params import args
import matplotlib.lines as mlines

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

def draw_multi(y, save=True, title=None, labels=None):
    """
    画多线图
    y: shape为(n, length)的numpy数组或类似结构，每一行为一条曲线
    labels: 每条曲线的标签，长度为n的列表
    """
    fig = plt.figure()
    n = y.shape[0]
    for i in range(n):
        if labels is not None:
            plt.plot(y[i], label=labels[i])
        else:
            plt.plot(y[i], label=f"line_{i+1}")
    if title is not None:
        plt.title(title)
    plt.legend()
    if save:
        fig.savefig('./checkpoint/fig/' + (title if title else "multi_line") + '.jpg')
    else:
        plt.show()
    plt.close(fig)

def draw_loss(y, save=True, title=None, labels=None):
    """
    画多线图
    y: shape为(n, length)的numpy数组或类似结构，每一行为一条曲线
    labels: 每条曲线的标签，长度为n的列表
    """
    fig = plt.figure()
    plt.xticks([])
    # 设置x轴和y轴的标签
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    n = y.shape[0]
    for i in range(n):
        if labels is not None:
            plt.plot(y[i], label=labels[i])
        else:
            plt.plot(y[i], label=f"line_{i+1}")
    if title is not None:
        plt.title(title)
    plt.legend()
    if save:
        fig.savefig('./checkpoint/fig/ablationBloss.pdf')
    else:
        plt.show()
    plt.close(fig)



def sub_figure(y, save=True, title=None, label=None):
    """
    >>> sub_figure(y=y, x_ax=x, title='test')
    """
    n, length = y.shape
    fig, ax = plt.subplots(figsize=(10, 6*n))  # 设置图片大小为10x6英寸
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


def draw_boxplot_by_your_husband(data_list, title, use_data, length):
    noise_range = ['raw', 1, -1, -3, -6]
    gcu_range = ['With GCU', 'Without GCU']
    if use_data == 'xjtu':
        colors = ['green', 'purple']
    elif use_data == 'mcc5':
        colors = ['blue', 'c']

    boxplot_positions = []
    colors_used = []
    boxplot_data = data_list

    for i, noise in enumerate(noise_range):
        for j, gcu in enumerate(gcu_range):
            boxplot_positions.append(i * (len(gcu_range)+1) + j + 1)
            colors_used.append(colors[j])

    # Create the boxplot
    fig, ax = plt.subplots(figsize=(10, 8))
    # Define properties for the boxplot
    boxprops = dict(linewidth=1, color='black')  # Box properties
    whiskerprops = dict(linewidth=1, color='black')  # Same color for whiskers
    meanprops = dict(marker='D', markeredgecolor='black', markerfacecolor='white', markersize=6,alpha = 0.8)  # Mean marker
    flierprops = dict(marker='o', markerfacecolor='white', markeredgecolor='black', markersize=6)  # Outliers as small hollow circles
    midprops = dict(linewidth=1, color='m',alpha=1)  # Middle line
    # Create the boxplot
    box = ax.boxplot(
        boxplot_data, positions=boxplot_positions, patch_artist=True, showfliers=True, showmeans=True,
        boxprops=boxprops, whiskerprops=whiskerprops, meanprops=meanprops, flierprops=flierprops, medianprops=midprops
    )

    # Set colors for the boxes (light fill, dark edge)
    for patch, color in zip(box['boxes'], colors_used):
        # patch.set_facecolor(color)  # Set box fill color
        rgba_color = plt.cm.colors.to_rgba(color, alpha=0.4)  # Choose a color and adjust the transparency
        patch.set_facecolor(rgba_color)
        patch.set_edgecolor('black')  # Set edge color for consistency



    # Customize the plot
    ax.set_xticks(
        ticks=[(i * (len(gcu_range)+1) + len(gcu_range) / 2 + 0.5) for i in range(len(noise_range))],
        labels=[f"{i}" for i in noise_range],
        fontsize=10
    )
    ax.set_xlabel("Noise", fontsize=22)
    ax.set_ylabel("Accuracy Rate", fontsize=22)
    ax.tick_params(axis='both', labelsize=20)
    ax.set_title(title, fontsize=24)
    
    # Create the legend handles for the flier, mean, and median
    flier_marker = mlines.Line2D([], [], marker='o', color='w', markerfacecolor='white', markeredgecolor='black', markersize=6, label='Flier')
    mean_marker = mlines.Line2D([], [], marker='D', color='w', markeredgecolor='black', markersize=6, label='Mean')
    median_line = mlines.Line2D([], [], color='m', label='Median')
    # Create the legend handles for the methods
    method_legend_handles = [plt.Line2D([0], [0], color=color, lw=4,alpha=0.4) for color in colors]
    method_legend_labels = gcu_range
    legend1 = ax.legend(
        handles=method_legend_handles,
        labels=method_legend_labels,
        title="Methods",
        loc="upper right",
        bbox_to_anchor=(1, 1),
        frameon=False,
        fontsize=10,
        title_fontsize=20
    )
    # legend1.set_title("Methods", prop={ 'weight': 'bold'})
    ax.legend(
        handles=[flier_marker, mean_marker, median_line],
        labels=['Flier', 'Mean', 'Median'],
        title="Markers",
        loc="upper right" ,
        bbox_to_anchor=(0.999, 0.88),
        frameon=False,
        fontsize=10,
        title_fontsize=20
    )
    ax.add_artist(legend1)
    plt.tight_layout()

    plt.savefig(f'./checkpoint/fig/ablitionA_{use_data}_{length}.pdf')
    plt.close()
    
    

def draw_boxplot(data_list, title, use_data, length):
    # 设置箱线图的位置，使组内相邻，组间有较大间隔
    positions = []
    group_gap = 2  # 组间间隔
    for i in range(5):
        positions.append(i * group_gap + 1)
        positions.append(i * group_gap + 1.7)

    # 设置每个箱线图的标签
    labels = []
    for snr in ['raw', 1, -1, -3, -6]:
        labels.append(f'{snr}')
        labels.append('')

    plt.figure(figsize=(16, 8))
    box = plt.boxplot(data_list, positions=positions, widths=0.6, patch_artist=True, showmeans=True)

    # 设置颜色，组内同色，组间不同色
    if use_data == 'xjtu':
        colors = ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c', '#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5']
    else:
        colors = ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c', '#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)

    plt.xticks(positions, labels, rotation=30)
    plt.title(title)
    plt.ylabel('准确率')
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(f'./checkpoint/fig/ablitionA_{use_data}_{length}.jpg')
    plt.close()


def draw_fill(series1: np.ndarray, series2: np.ndarray, label1: str, label2: str, title: str, save_path: str) -> None:
    """
    绘制两条曲线，并在每条曲线与y=0之间填充颜色，带有'.'标记。
    图像等比放大，避免画面太挤。
    设置横坐标标签为Training Steps，纵轴标签为acc。
    画面于边框之间不要有空白。
    """
    x = np.arange(series1.size)
    fig, ax = plt.subplots(figsize=(8, 6))  # 等比放大图像
    c1, c11 = '#8dee5c', '#97d65b'
    c2, c22 = '#e45a41', '#e67860'
    ax.set_xlabel('Training Steps', fontname='Times New Roman', fontsize=12)
    ax.set_ylabel('Gradient Norm', fontname='Times New Roman', fontsize=12)
    # 不显示x轴的坐标
    ax.set_xticks([])

    # 第一条曲线：线+带80%透明度的'.'标记+填充到y=0
    ax.plot(x, series1, color=c1, marker='*', markersize=3, linewidth=0.1, label=label1, alpha=1)
    ax.fill_between(x, series1, 0, color=c11, alpha=0.2)

    # 第二条曲线：线+带80%透明度的'.'标记+填充到y=0
    ax.plot(x, series2, color=c2, marker='.', markersize=3, linewidth=0.1, label=label2, alpha=.9)
    ax.fill_between(x, series2, 0, color=c22, alpha=0.2)

    # 基线y=0
    ax.axhline(0, color='k', linewidth=0.8, alpha=0.3)

    if title:
        ax.set_title(title)
    ax.legend(fontsize=12)

    # 去除画面与边框之间的空白
    # plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(f'./checkpoint/fig/ablitionB.jpg', bbox_inches='tight')
    plt.close(fig)


# ****************************** About Data **********************************
def get_info(x: np.ndarray) -> np.ndarray:
    """
    x: np.ndarray
    """
    max, min = np.max(x, axis=1), np.min(x, axis=1)
    mean, std = np.mean(x, axis=1), np.std(x, axis=1)
    return np.array([max, min, mean, std])

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

    def forward(self, x, alpha=1):
        b, c, l = x.shape
        p = x.clone()
        if args.blocker:
            p = GradBlocker.apply(p, 0)
        return x * self.se(p).view(b, c, 1)

def gcu(x):
    c = x.shape[1]
    ffted = torch.fft.fft(x)
    yi, yr = torch.abs(ffted.imag), torch.abs(ffted.real)
    return yi, yr   # (batch, c, l)

def fft(x):
    ffted = torch.fft.fft(x)
    return torch.abs(ffted), torch.abs(ffted)   # (batch, c, l)

def no_gcu(x):
    return x, x

class Transfer(nn.Module):
    def __init__(self, fft_type) -> None:
        super().__init__()
        if fft_type == 'gcu':
            self.transfer = gcu
        elif fft_type == 'fft':
            self.transfer = fft
        else:
            self.transfer = no_gcu
        
    def forward(self, x):
        return self.transfer(x)

if __name__ == '__main__':
    x = torch.randn(1, 6, 100)
    y = gcu(x)
    print(y.shape)

