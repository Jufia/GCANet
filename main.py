import torch
from torch import nn
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import random
import math

import sys

from models import model_dic
from Process import loador_dict
from params import args
import utilise


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


device = args.GPU
loss_all = []
acc_all = []
acct_all = []


def evaluate_model(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    acc = []
    with torch.no_grad():
        for inputs, labels in data_loader:
            if args.snr != None:
                inputs += utilise.wgn(inputs, args.snr)
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels.view(-1)).sum().item()
            acc.append(correct / labels.size(0))

        # condidton = predicted != labels
        # print('  pre: {}\n  tag: {}'.format(predicted[condidton], labels[condidton].type(torch.LongTensor)))
        # logging.info(f"  pre: {predicted[condidton]}\n  tag: {labels[condidton].type(torch.LongTensor)}")
    return np.mean(acc)*100, np.std(acc)*100

def L2_regulation(weight, lambda_l2):
    return lambda_l2 * torch.norm(weight, p=2)


def train(model, train_loader, valid_loder, test_loader):
    total_step = len(train_loader)
    logging.info(f"total step = {total_step}")
    criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.lambda_l2)
    optimizer = torch.optim.SGD([
        {'params': model.parameters()},
    ], lr=args.lr, weight_decay=args.lambda_l2)
    # optimizer = torch.optim.SGD([{'params': model.parameters()}, {'params': model.classfier}], lr=args.lr)
    scheduler = MultiStepLR(optimizer, milestones=[20, 40, 60, 80], gamma=args.lr_decay)  # learning rates
    for epoch in range(args.epochs):
        model = model.to(device)
        
        loss_sum = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            p = (epoch*total_step + batch_idx) / (args.epochs * total_step)
            alpha = 2. / (1. + math.exp(-5 * p)) - 1
            model.train()
            if args.snr != None:
                data += utilise.wgn(data, args.snr)
            data, target = data.to(device), target.type(torch.LongTensor).to(device)
            if args.algorithm == 'GCA':
                pre = model(data, alpha*1)
            else:
                pre = model(data)

            optimizer.zero_grad()
            loss = criterion(pre, target.view(-1))
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            if (batch_idx + 1) % (total_step//10) == (total_step//10 - 1):
                train_acc, stdt = evaluate_model(model, train_loader)
                correct, stdv = evaluate_model(model, valid_loder)
                test_acc, std = evaluate_model(model, test_loader)

                acc_all.append(train_acc)
                acct_all.append(test_acc)
                loss_all.append(loss_sum / (batch_idx + 1))

                logging.info(
                    f"Epoch [{epoch + 1}/{args.epochs}], Step[{batch_idx + 1}/{total_step}], alpha={alpha} Loss: {loss.item()} TrainACC: {train_acc}%, ValidationAcc: {correct}%, TestACC: {test_acc}, STDtrain: {stdt}, STDvalid: {stdv}, STDtest: {std}")
                
                if correct > args.best_model:
                    args.best_model = correct
                    if args.algorithm == 'GCAL':
                        name = utilise.get_name(args, correct)
                        torch.save(model.state_dict(), name)
                        if args.best_model == 100:
                            print(f"{args.log_name} is achieve 100%, program over!")
                            sys.exit()


def test():
    model = model_dic[args.algorithm]()
    train_loader, valid_loader, test_loader = loador_dict[args.use_data]()
    total = sum([param.nelement() for param in model.parameters()])
    print('***************** Number of parameter: {} Thousands ********************'.format(total / 1e3))
    logging.info(f"***************** Number of parameter: {total / 1e3} Thousands ********************")
    logging.info(f"setting: {args}")
    train(model, train_loader, valid_loader, test_loader)
    logging.info(f"*********The hightest ACC is {args.best_model}*************")
    # np.save('./checkpoint/fig/'+args.log_name+'.npy', np.array([loss_all, acc_all, acct_all]))
    utilise.draw(loss_all, title=args.log_name+'loss')
    utilise.sub_figure(np.array([acc_all, acct_all]), title=args.log_name)
    # acc = evaluate_model(model, test_loader) logging.info(f"Use FFC: {args.ffc} & Use Attention: {args.att} The Acc
    # on dataset {args.use_data} is {acc * 100:.2f}%")


if __name__ == '__main__':
    test()
