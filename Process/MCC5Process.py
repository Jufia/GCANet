"""
Load MCC5-THU dataset
"""
import numpy as np
import torch
import os
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

from params import args
from utilise import *

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


# def remove_element_2(lst, target):
#     return list(filter(lambda x: target not in x, lst))

# def rename():
#     files = os.listdir(args.path)
#     os.chdir(args.path)
#     aa = [n for n in files if 'teeth_break_' in n]
#     old = remove_element_2(aa, 'and')
#     end = 11
#     new = [name[:end]+'_single'+name[end:] for name in old]
#     for o, n in zip(old, new):
#         print(o)
#         print(n)
#         os.rename(o, n)

def get_mcc5(root):
    files = os.listdir(root)
    # status = ['health', 'miss_teeth', 'gear_wear', 'gear_pitting', 'teeth_crack', 'teeth_break_single',
    #   'teeth_break_and_bearing_inner', 'teeth_break_and_bearing_outer']
    status = ['health', 'miss_teeth', 'gear_wear_H', 'gear_pitting_H', 'teeth_crack_H', 'teeth_break_single_H',
              'teeth_break_and_bearing_inner_H', 'teeth_break_and_bearing_outer_H']
    statu_path = []
    for statu in status:
        name = [na for na in files if statu in na]
        statu_path.append(name)

    file_dict = {i: value for i, value in enumerate(statu_path)}
    return file_dict


def mcc5_read(file, label):
    data = pd.read_csv(file).iloc[:, :8]
    # result = np.array([data['torque'], data['motor_vibration_x'], data['gearbox_vibration_x']])
    # result = np.array([data['gearbox_vibration_x'], data['gearbox_vibration_y'], data['gearbox_vibration_z']])
    result = np.array([value for key, value in data.items()])
    result = torch.Tensor(result).squeeze()

    window = args.windows
    endpoint = result.shape[1]
    length = args.length

    samples = torch.empty([0, result.shape[0], length])
    for start in range(0, endpoint - length, window):
        ss = result[:, start:start + length]
        ss = torch.FloatTensor(ss).reshape(1, -1, args.length)
        samples = torch.cat((samples, ss), dim=0)

    # samples = min_max_normalize(samples.view(-1, result.shape[0], args.length))
    num = samples.shape[0]
    logging.info(f"label: {label} total {num}")
    labels = label * torch.ones(num)

    return samples, labels


def MCC5_Merge_Save():
    # root = '/document/U/language/DataHub/MCC5_THU/'
    root = args.path
    dataname_dict = get_mcc5(root)

    samples = torch.empty([0, 8, args.length])
    args.in_channel = 8
    labels = torch.empty(0, dtype=torch.float)
    for label, file_names in dataname_dict.items():
        for file_name in file_names:
            print(label, file_name)
            file = os.path.join(root, file_name)
            sub_samples, sub_labels = mcc5_read(file, label)
            samples = torch.cat((samples, sub_samples), dim=0)
            labels = torch.cat((labels, sub_labels), dim=0)

    torch.save(samples, './data/mcc5/unoverleap512x.pth')
    torch.save(labels, './data/mcc5/unoverleap512y.pth')
    return samples, labels


def Loador():
    # x, y = MCC5_Merge_Save()
    x = torch.load('./data/mcc5/unoverleap512x.pth', weights_only=False)  # (batch, 8, m, m)
    y = torch.load('./data/mcc5/unoverleap512y.pth', weights_only=False)
    # x = torch.load('./data/MCC5_THU/data_samples36_fft.pth', weights_only=False)
    # y = torch.load('./data/MCC5_THU/data_labels36_fft.pth', weights_only=False)

    args.in_channel = x.shape[1]
    args.class_num = len(torch.unique(y))
    x_train, x_remain, y_train, y_remain = train_test_split(x, y, test_size=0.3, shuffle=True, random_state=44)
    x_valid, x_test, y_valid, y_test = train_test_split(x_remain, y_remain, test_size=0.5, shuffle=False, random_state=1)

    train_set = TensorDataset(x_train, y_train)
    valid_set = TensorDataset(x_valid, y_valid)
    test_set = TensorDataset(x_test, y_test)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader


if __name__ == '__main__':
    # rename()
    Loador()
