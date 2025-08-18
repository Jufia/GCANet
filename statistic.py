'''
python -i statistic.py
'''
import torch
from thop import profile,clever_format
import time 
from params import args
# from torchsummary import summary
from torchinfo import summary

from models import model_dic

"""
model_dic = {
                'GCA': GCANet,
                'PatchTST': patchtstModel,
                'softshape': SoftShapeNet,
                'TimesNet': TimesNet,
                'ModernTCN': ModernTCNd,
                'TimeMixer': TimeMixer,
                'DLinear': dlinear,
                'LightTS': LightTS,
                'InceptionTime': InceptionTime,
                'TCN': TCN,
             }
"""

def quality(model_name):
    model = model_dic[model_name]().to(torch.device(args.GPU))
    # summary(model,(args.in_channel, args.length), device='cuda')
    # summary函数会显示模型参数总数，但不会直接显示模型占用的内存大小（如MB/GB）。
    # 下面代码可以估算模型参数占用的显存（仅参数，不含中间激活/缓存）：
    summary(model, input_size=(args.batch_size, args.in_channel, args.length), device=args.GPU)

    # 计算模型参数占用的内存（单位：MB）
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024**2
    print(f"model size is {size_all_mb:.3f} MB")

    x = torch.rand(args.batch_size, args.in_channel, args.length).to(args.GPU)
    model = model.to(args.GPU)
    start_time = time.time()
    out = model(x)
    end_time = time.time()
    flops, params = profile(model, inputs=(x,))

    # conver results to be legible
    f, p = clever_format([flops, params], '%.3f')
 
    print(f"***************** Detales of model {model_name} Input shape is {x.shape} **********************")
    print(f"Flops of model: {f}  ({flops}), \n  number of parameters: {p} ({params}), \n Time to calculate: {(end_time-start_time)*1000} ms \n")

if __name__ == '__main__':
    quality('GCA')


    # x1 = torch.arange(1, 4).reshape(1, 3).unsqueeze(-1) * torch.ones(1, 3, 3)
    # x2 = torch.arange(11, 14).reshape(1, 3).unsqueeze(-1) * torch.ones(1, 3, 3)
    # x = torch.cat((x1, x2), dim=1)
    # m1 = torch.nn.Conv1d(6, 6, kernel_size=1, groups=2, bias=False)
    # m2 = torch.nn.Conv1d(6, 6, kernel_size=1, groups=3, bias=False)
    # torch.nn.init.ones_(m1.weight)
    # torch.nn.init.ones_(m2.weight)
    # y1 = m1(x)
    # y2 = m2(x)
    # m2.named_parameters
    
    # x = torch.rand(512, 6, 1024).transpose(1,2)
    # m = torch.nn.RNN(input_size=6, hidden_size=6, batch_first=True, num_layers=2)