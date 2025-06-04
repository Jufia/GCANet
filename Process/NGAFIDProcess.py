import pandas as pd
import numpy as np
import os
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

from params import args
from statistic import count
import logging
logging.basicConfig(
     filename=args.log_name,
     encoding="utf-8",
     filemode="w",
     format="{asctime} - {levelname} - {message}",
     style="{",
     datefmt="%Y-%m-%d %H:%M",
     level=logging.DEBUG,
)


def fft_tsfer(img: np.ndarray, *args, **kwargs) -> torch.FloatTensor:
    dft = np.fft.fft2(img)
    # fft_shift = np.fft.fftshift(dft, axes=(1, 2, 3))
    # log_fft_shift = np.log(fft_shift)
    fft_shift_abs = np.abs(dft)

    return torch.FloatTensor(fft_shift_abs)


def single_data(time_seri: np.array):
    # (length, 23) --> (batch, 23, m, m)
    
    window = args.windows
    length = pow(args.m, 2)
    endpoint, channel = time_seri.shape
    data = time_seri.transpose()
    feature = data

    samples = torch.empty([0, channel, args.m, args.m], dtype=torch.float)
    for start in range(0, endpoint - length, window):
        ss = data[:, start:start+length]
        ss = fft_tsfer(ss.reshape(1, channel, args.m, args.m))
        samples = torch.cat((samples, ss), axis=0)
    
    return samples


def re_label(label: np.array) -> np.array:
    resort = np.zeros(label.shape)
    for i, orign in enumerate(label):
        unq = np.unique(orign)
        tag = np.arange(len(unq))
        for a, b in zip(unq, tag):
            resort[i][orign == a] = b

    print('label:', label, 'resort:', resort)
    logging.info(f"label: {label}, resort: {resort}")
    return resort


def ngafid_read():
    # path = '/document/U/language/DataHub/NGAFID/2days/'
    path = args.path
    flight_header = pd.read_csv(path+'flight_header.csv', index_col='Master Index')
    flight_data = pd.read_pickle(path+'flight_data.pkl')

    samples = torch.empty([0, 23, args.m, args.m], dtype=torch.float)
    labels = np.empty([3, 0])
    count = 0
    logging.debug("this is 【index】, number of nan, sum of nan")
    for index, header in flight_header.iterrows():
        data = flight_data[index]
        if len(data) < 4096:
            count += 1
            continue
        logging.info(f"【{index}】: dim0: {sum(np.isnan(data))} dim1: {sum(np.isnan(data.transpose()))} "
                     f"sum(): {np.sum(np.isnan(data), keepdims=False)}")
        data[np.isnan(data)] = 0
        sub_sample = single_data(data)
        label, tlabel, hlabel = header['class'], header['target_class'], header['hclass']
        samples = torch.cat([samples, sub_sample], dim=0)
        sub_label = np.ones([len(sub_sample), 3]) * np.array([label, tlabel, hlabel])
        labels = np.concatenate((labels, sub_label.transpose()), axis=1)

    logging.debug(f"**there are {count} data < 4096 **")
    if not os.path.exists('./data/NGAFID'):
        os.mkdir('./data/NGAFID')

    labels = re_label(labels)
    torch.save(samples, './data/NGAFID/samples36_no_fft_shift.pth')
    torch.save(torch.Tensor(labels), './data/NGAFID/labels36_no_fft_shift.pth')
    return samples, torch.Tensor(labels)


def Loador():
    x, y = ngafid_read()
    # x = torch.load('./data/NGAFID/samples36.pth', weights_only=False)
    # y = torch.load('./data/NGAFID/labels36.pth', weights_only=False)
    y = y[0]
    args.class_num, args.class_count = count(data=y)
    x_train, x_test, y_train, y_test = train_test_split(x, torch.Tensor(y), test_size=0.2)
    train_set = TensorDataset(x_train, y_train)
    test_set = TensorDataset(x_test, y_test)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True)

    return train_loader, test_loader


if __name__ == '__main__':
    # train, test = Loador()
    train, test = ngafid_read()
