import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

from params import args
import logging

logging.basicConfig(
    filename='./checkpoint/log/' + args.log_name + '.log',
    encoding="utf-8",
    filemode="w",
    format="{asctime} - {levelname} - {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M",
    level=logging.DEBUG,
)


def single_data(data: np.array, lable: int):
    # (channel, lenght) -> (batch, channel, m, m)
    window = args.length
    length = args.length
    channel, endpoint = data.shape

    samples = torch.empty([0, 2, args.length], dtype=torch.float)
    for start in range(0, endpoint - length, window):
        ss = data[:, start:start + length]
        ss = torch.FloatTensor(ss).reshape(1, -1, args.length)
        samples = torch.cat((samples, ss), axis=0)

    samples_num = len(samples)  # (n, channel, m^2)
    labels = np.full(samples_num, lable)
    logging.info(f"label: {lable} count: {samples_num}")

    return samples, labels


def xjtu_read():
    # path = '/document/U/language/DataHub/XJTU/XJTU_Gearbox/'
    path = args.path
    files = os.listdir(path)

    samples = torch.empty([0, 2, args.length], dtype=torch.float)
    labels = np.empty(0)
    for i, file in enumerate(files):
        y_vibration = np.loadtxt(os.path.join(path, file, 'Chan1.txt'))
        x_vibration = np.loadtxt(os.path.join(path, file, 'Chan2.txt'))

        data = np.array([x_vibration, y_vibration])
        sample, label = single_data(data, i)
        samples = torch.cat((samples, sample), axis=0)
        labels = np.append(labels, label)

    if not os.path.exists('./data/XJTU'):
         os.mkdir('./data/XJTU')
    torch.save(samples, './data/XJTU/unoverleap512x.pth')
    torch.save(torch.Tensor(labels), './data/XJTU/unoverleap512y.pth')
    return samples, torch.Tensor(labels)


def Loador():
    # x, y = xjtu_read()
    x = torch.load(f'{args.path}unoverleap{args.length}x.pth', weights_only=False)
    y = torch.load(f'{args.path}unoverleap{args.length}y.pth', weights_only=False)
    args.in_channel = x.shape[1]
    args.class_num = len(torch.unique(y))
    x_train, x_test, y_train, y_test = train_test_split(x, torch.Tensor(y), test_size=0.4, random_state=args.random_state)
    x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5, random_state=args.random_state)
    train_set = TensorDataset(x_train, y_train)
    test_set = TensorDataset(x_test, y_test)
    val_set = TensorDataset(x_val, y_val)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    logging.info(f"total train samples: {len(train_set)} times batch")
    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    t, s = xjtu_read()
    # train, test = Loador()
