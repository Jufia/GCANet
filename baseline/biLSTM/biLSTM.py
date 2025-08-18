import torch
import torch.nn as nn

from params import args

class biLSTM(nn.Module):
    def __init__(self):
        super(biLSTM, self).__init__()
        self.input_size = args.in_channel  # 输入通道数
        self.seq_len = args.length         # 序列长度
        self.hidden_size = 64              # 隐藏层维度，可根据需要调整
        self.num_layers = 2                # LSTM层数，可根据需要调整
        self.num_classes = args.class_num  # 分类类别数

        # LSTM: 输入(batch, seq_len, input_size)
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True
        )
        # 分类头
        self.fc = nn.Linear(self.hidden_size * 2, self.num_classes)

    def forward(self, x):
        # x: (batch, channels, length)
        # 转换为(batch, length, channels)
        x = x.permute(0, 2, 1)
        # LSTM
        out, _ = self.lstm(x)  # out: (batch, length, hidden_size*2)
        # 取最后一个时间步的输出
        out = out[:, -1, :]    # (batch, hidden_size*2)
        out = self.fc(out)     # (batch, num_classes)
        return out


if __name__ == "__main__":
    model = biLSTM()
    x = torch.randn(1, args.in_channel, args.length)
    print(model(x).shape)
