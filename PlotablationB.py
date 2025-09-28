from os import path
import numpy as np
from utilise import draw_loss, draw_fill

def avarage(seq, k):
    n = len(seq)
    res = []
    for i in range(n-k):
        res.append(np.mean(seq[i:i+k]))
    
    return np.array(res)


"""
loss.npy: [loss, train_acc, test_acc]
"""
def convergency(seed, type):
    epsilon = 1e-4
    for t in type:
        loss = []
        for s in seed:
            p = f'checkpoint/res/ablationB{t}_{s}_loss.npy'
            a = np.load(p)[0]
            loss.append(a)
        loss = np.array(loss)
        loss = np.mean(loss, axis=0)
        count, i = 0, 0
        while count < 5 and i < len(loss)-1:
            if loss[i] - loss[i+1] > epsilon:
                count = -1
            
            i += 1
            count += 1

        print(f'type [{t}] convergency at {i-count}_th step')

def plotACC(s, t, k, rank):
    acc = []
    for type in t:
        ac = []
        for seed in s:
            # p = f'checkpoint/res/ablationB_gcu_lr_1_rand_{seed}_{type}_loss.npy'
            p = f'checkpoint/res/ablationB{type}_{seed}_loss.npy'
            a = np.load(p)[rank][:100]
            a = avarage(a, k)
            ac.append(a)
        ac = np.array(ac)
        ac = np.mean(ac, axis=0)
        acc.append(a)

    draw_loss(np.array(acc), title=None)
    print(f'complete random {seed} k={k}')

def plotSTD(s, t, k):
    std = []
    for type in t:
        d = []
        for seed in s:
            p = f'checkpoint/res/ablationB{type}_{seed}_grad.npy'
            a = np.load(p)
            a = a.reshape(1, -1).squeeze(0)
            # a = np.std(a, axis=1)
            d.append(a)

        d = np.array(d)
        d = np.mean(d, axis=0)
        std.append(d)
    # draw_multi(np.array(std), title=f'random_{s}')
    draw_fill(std[1], std[0], 'without GGL', 'with GGL', title=None, save_path=None)
    print('done')


if __name__ == '__main__':
    # plotACC([42, 3470, 215], [1, 2], 1, 0)
    plotSTD([42, 3470, 215], [1, 2], 1)
    # convergency([42, 3470, 215], [1, 2, 3])