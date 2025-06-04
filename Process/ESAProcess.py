"""
Import original data and transform it to {time, values, label} table
function:
unite_time(): 'StartTime'&'EndTime' in label.csv using time form yy-mm-ddThh:mm:ssZ while serching mathod in
    channelxx.csv using yy-mm-dd hh:mm:ss
generate_unique_label_csv(): 'StartTime' and 'EndTime' repeated a lot, generate a .csv file which has more clear data
    to make lable more convinient just run it once to generate the 'unique_labels.csv' file
time_span(): a function to sumerize events
count_label(): a function to count the number of labels in each event

add_label(): a function to add label to data
change_label(): a function to change the label of data according to event and time
fill_loss_label(): a function to fill the loss label
save channelxx.zip keys(): time value label
"""
import pandas as pd
import numpy as np
import os
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

from params import args

# d42 = pd.read_pickle(os.path.join(path, 'channel_42.zip'))
# d43 = pd.read_pickle(os.path.join(path, 'channel_43.zip'))
# d44 = pd.read_pickle(os.path.join(path, 'channel_44.zip'))
# d45 = pd.read_pickle(os.path.join(path, 'channel_45.zip'))
# d46 = pd.read_pickle(os.path.join(path, 'channel_46.zip'))


#%%
def unite_time(time: str):
    '''
    2005-01-17T03:22:45.423Z -> 2005-01-17 03:22:45.423
    '''
    day = time.split('T')[0]
    moment = time.split('T')[1][:-1]
    return day + ' ' + moment


def generate_unique_label_csv():
    '''
    'StartTime' and 'EndTime' repeated a lot,
    generate a .csv file which has more clear data to make lable more convinient
    just run it once to generate the 'unique_labels.csv' file
    '''
    event = pd.read_csv('/document/U/language/DataHub/ESA_AD/Mission1/labels.csv')
    label_time = event.filter(items=['ID', 'StartTime', 'EndTime'])
    unique_label_time = label_time.drop_duplicates()
    unique_label_time.to_csv('/document/U/language/DataHub/ESA_AD/Mission1/unique_label_time.csv', index=False)


def time_span(label: np.array):
    label = label.reshape(-1)
    time_points = []
    lb = []
    for i in range(label.shape[0]-1):
        if label[i+1]-label[i] != 0:
            time_points.append(i+1)
            lb.append(label[i])
    span = [time_points[0]]
    for i in range(len(time_points)-1):
        span.append(time_points[i+1]-time_points[i])

    s = {'label': lb, 'span': span}
    df = pd.DataFrame(s)
    df.to_csv('./data/ESA/time_span.csv')


def count_label(label: np.array):
    count = np.bincount(label)
    s = {'label': np.arange(label.max()+1), 'count': count}
    df = pd.DataFrame(s)
    df.to_csv('./data/ESA/count_label.csv', index=False)


def add_label_to_format(data_tabel: pd.DataFrame):
    label = np.zeros([data_tabel.shape[0], 1]).astype(int)
    data_tabel['label'] = label
    return data_tabel


def change_label(data_tabel: pd.DataFrame):
    event = pd.read_csv('/document/U/language/DataHub/ESA_AD/Mission1/unique_label_time.csv')
    # event.keys() = Index(['ID', 'StartTime', 'EndTime'], dtype='object')
    event_type = pd.read_csv('/document/U/language/DataHub/ESA_AD/Mission1/anomaly_types.csv')
    # event_type.keys() = Index(['ID', 'Class', 'Subclass', 'Category', 'Dimensionality', 'Locality', 'Length'],
    #       dtype='object')
    rare_event_id = event_type.loc[event_type['Category'] == 'Rare Event', 'ID']
    for i in range(event.shape[0]):
        if event.iloc[i]['ID'] in list(rare_event_id):
            # 78个 rare event 不算故障，排除它们
            continue
        LB, ST, ET = event.iloc[i]
        lb = int(LB.split('_')[1])
        st = unite_time(ST)
        et = unite_time(ET)
        data_tabel.loc[st:et, 'label'] = lb

    return np.float32(data_tabel)


def ESA_process():
    data = pd.read_pickle('/document/U/language/DataHub/ESA_AD/Mission1/channels/channel_41.zip')
    data = add_label_to_format(data)
    data = change_label(data)
    data.to_pickle('./data/ESA/channel41.zip')


#%%
def fill_loss_label(data_tabel: np.array):
    # array([  7,  15,  16,  26,  34,  41,  42,  44,  55, 144])
    loss_label = np.setdiff1d(np.arange(201), np.unique(data_tabel))
    limit = pd.unique(data_tabel).shape[0]
    loss_label = loss_label[loss_label < limit]
    for new in loss_label:
        old = np.unique(data_tabel).max()
        data_tabel[data_tabel==old] = new

    # data_tabel.to_pickle('./data/ESA/channel41.zip')
    return data_tabel, limit


def min_max_normalize(data: torch.FloatTensor):
    data_max, data_min = data.max(dim=1).values, data.min(dim=1).values
    # data_norm = (data - data_min.view(3, -1)) / (data_max - data_min + 1e-6).view(3, -1)
    data_norm = (data - data_min.view(1, -1)) / (data_max - data_min + 1e-6).view(1, -1)
    return data_norm


def set_label(lb: np.array):
    purity = args.purity * len(lb)
    value = np.unique(lb)
    if len(value) == 1:
        label = value[0]
    else:
        label = value[0] if sum(lb == value[0]) > purity else value[1]

    return int(label)


def esa_read():
    data = pd.read_pickle(args.path)
    data = np.array(data).transpose()
    vl, lb = data[0], data[1]
    result = torch.Tensor(vl).unsqueeze(0)

    window = args.windows
    endpoint = data.shape[1]
    length = pow(args.m, 2)

    samples = torch.empty([0, 3, args.m, args.m], dtype=torch.float)
    labels = np.empty(0).astype(int)
    for start in range(0, endpoint - length, window):
        sub_label = set_label(lb[start:start + length])
        if ((sub_label == 0) & (torch.rand(1).item() < args.drop_zero)):
            continue

        labels = np.append(labels, sub_label)
        sub_sample = result[:, start:start + length]
        sub_sample = min_max_normalize(sub_sample)
        samples = torch.cat((samples, sub_sample.repeat(3, 1).reshape(1, 3, args.m, args.m)), dim=0)

    torch.save(samples, './data/ESA/sample.pth')
    torch.save(labels, './data/ESA/label.pth')
    return samples, labels


def Loador():
    # x, y = esa_read()
    x = torch.load('./data/ESA/samples.pth')
    y = torch.load('./data/ESA/labels.pth')
    y, class_number = fill_loss_label(np.array(y))
    x_train, x_test, y_train, y_test = train_test_split(x, torch.tensor(y), test_size=0.2)
    train_set = TensorDataset(x_train, y_train)
    test_set = TensorDataset(x_test, y_test)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True)

    return train_loader, test_loader, class_number


if __name__ == '__main__':
    train_l, test_l = Loador()
