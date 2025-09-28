"""
ablationA: 画箱式图
"""
from pathlib import Path
from Process import loador_dict
from models.GCAL import GCANet
from params import args
import torch
import numpy as np
from utilise import draw_boxplot, draw_boxplot_by_your_husband, wgn

def evaluate_model(model, data_loader):
    acc = []
    with torch.no_grad():
        for inputs, labels in data_loader:
            if args.snr != 'raw':
                inputs += wgn(inputs, args.snr)
            inputs, labels = inputs.to(args.GPU), labels.to(args.GPU)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels.view(-1)).sum().item()
            acc.append(correct / labels.size(0))
    
    return acc

def get_data(data, length, GPU, head):
    args.length = length
    args.use_data = data
    args.GPU = GPU
    args.head = head

    for gcu in ['gcu', 'nogcu']:
        args.gcug = gcu if gcu == 'gcu' else 'none'
        model = GCANet().to(args.GPU)
        for args.snr in [-1, -3]:
            model_weight = Path('./checkpoint/dic/', f'ablitionA_{args.length}_{args.use_data}_{args.snr}_{gcu}.log.pth')
            model.load_state_dict(torch.load(model_weight, weights_only=True))
            model.eval()

            _, _, test_loader = loador_dict[args.use_data]()
            acc = evaluate_model(model, test_loader)

            save_path = Path('./checkpoint/res', f'ablitionA_{args.length}_{args.use_data}_{args.snr}_{gcu}.npy')
            np.save(save_path, np.array(acc))
            print(f'{args.use_data}_{args.length}_{args.snr}_{gcu} done')

def plot_data(data, length):
    data = data
    length = length
    data_list = []
    for snr in ['raw', 1, -1, -3, -6]:
        gcu = np.load(Path('./checkpoint/res', f'ablitionA_{length}_{data}_{snr}_gcu.npy'))
        data_list.append(gcu)
        nogcu = np.load(Path('./checkpoint/res', f'ablitionA_{length}_{data}_{snr}_nogcu.npy'))
        data_list.append(nogcu)
    name = {'xjtu': 'XJTUGearbox', 'mcc5': 'MCC5-THU'}
    title = f'{name[data]} Accuracy for Length {length}'
    draw_boxplot_by_your_husband(data_list, title, data, length)
    print(f'plot {data}_{length} done')

    
if __name__ == '__main__':
    data = 'mcc5'
    length = 1024
    GPU = 'cuda:1'
    head = 6
    # get_data(data, length, GPU, head)
    for data, length in [('xjtu', 1024), ('mcc5', 1024), ('xjtu', 512), ('mcc5', 512)]:
    # for data, length in [('xjtu', 1024)]:
        plot_data(data, length)
