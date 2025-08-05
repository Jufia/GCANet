import torch
import torch.nn as nn
from utilise import GradBlocker

import numpy as np
import random
import os

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # 保证多卡和cudnn的可复现性
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def _weights_init(m):
    if isinstance(m, nn.Conv1d):
        torch.nn.init.kaiming_uniform_(m.weight)
        # torch.nn.init.ones_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm1d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        torch.nn.init.kaiming_uniform_(m.weight)
        # torch.nn.init.ones_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

class Net(nn.Module):
    def __init__(self, input_dim, output_dim, channel):
        super(Net, self).__init__()
        set_seed(42)
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.att = nn.Linear(output_dim, 1)
        self.classifier = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(output_dim*channel, 5),
            nn.Sigmoid()
        )
        self.apply(_weights_init)

    def forward(self, x):
        x = self.fc1(x)

        p = x.clone()
        p = GradBlocker.apply(p, 0)
        w = self.att(p)
        y = x * w
        cls = self.classifier(y)
        cls = GradBlocker.apply(cls, 0.5)

        return cls
    
    def compute_gradient_norm(self):
        for p in self.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                print(param_norm)

def main():
    x = torch.ones(3, 8, 15)
    y = torch.LongTensor([0, 1, 2])
    m = Net(15, 9, 8)
    print(m.fc1.weight)
    hat = m(x)
    print(hat.shape)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(m.parameters(), lr=0.001)

    optimizer.zero_grad()
    loss = criterion(hat, y)
    loss.backward()
    m.compute_gradient_norm()
    optimizer.step()

    print(loss)

if __name__ == '__main__':
    main()
    pass