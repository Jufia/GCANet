import os
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

from params import args
import logging

logging.basicConfig(
    filename='./checkpoint/log/' + args.log_name,
    encoding="utf-8",
    filemode="w",
    format="{asctime} - {levelname} - {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M",
    level=logging.DEBUG,
)


def hit_read(path: str, classify_label: int):
    data = np.load(path)  # (504, 8, 20480)
    feature = data[:, 0:6, :]  # (504, 6, 20480)

    args.in_channel = feature.shape[1]
    window = args.windows
    endpoint = data.shape[-1]
    length = args.length

    samples = torch.empty([0, args.in_channel, length])
    for start in range(0, endpoint - length, window):
        ss = feature[:, :, start: start + length]
        ss = torch.FloatTensor(ss)
        samples = torch.cat((samples, ss), axis=0)

    samples_num = len(samples)  # (n, 6, 1024)
    labels = np.full(samples_num, classify_label)

    return samples, labels


def HIT_Merge_Save():
    # root = '/document/U/language/DataHub/HIT'
    root = args.path
    dataname_dict = ['data1.npy', 'data2.npy', 'data3.npy', 'data4.npy', 'data5.npy']
    label_set = [0, 0, 2, 3, 1]

    samples = torch.empty([0, 6, args.length])
    labels = np.empty(0)
    for i, filename in enumerate(dataname_dict):
        path = os.path.join(root, filename)
        classfy_label = label_set[i]
        sub_samples, sub_labels = hit_read(path, classfy_label)
        samples = torch.cat((samples, sub_samples), axis=0)
        labels = np.append(labels, sub_labels)

    if not os.path.exists(root):
        os.mkdir(root)
    torch.save(samples, './data/hit/unoverleap512x.pth')
    torch.save(torch.Tensor(labels), './data/hit/unoverleap512y.pth')
    return samples, torch.Tensor(labels)


def Loador():
    # x, y = HIT_Merge_Save()
    x = torch.load('./data/hit/unoverleap512x.pth', weights_only=False)  # (batch, 6, l)
    y = torch.load('./data/hit/unoverleap512y.pth', weights_only=False)
    args.in_channel = x.shape[1]
    args.class_num = len(torch.unique(y))

    # 如果给train_test_split设置random_state参数（即随机种子），每次划分的数据集都会是固定的
    x_train, x_test, y_train, y_test = train_test_split(x, torch.Tensor(y), test_size=0.4, random_state=42)
    x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5, random_state=42)

    train_set = TensorDataset(x_train, y_train)
    test_set = TensorDataset(x_test, y_test)
    val_set = TensorDataset(x_val, y_val)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, worker_init_fn=lambda worker_id: np.random.seed(42), generator=torch.Generator().manual_seed(42))
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=True)

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    train, test = Loador()
