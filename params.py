import argparse
import pandas as pd


layers = [
    [10, 24, 15, 2, "RE", True, True, 2, 24],
]

data_select = {'name': ['hit', 'xjtu', 'mcc5', 'dirg'],
               'path': ['/home/users/wzr/project/predict/Drive/ResNet50/data/hit/', 
                        '/home/users/wzr/project/predict/Drive/ResNet50/data/XJTU/XJTU_Gearbox/', 
                        '/home/users/wzr/project/predict/Drive/ResNet50/data/MCC5_THU/',
                        './data/dirg/'],
               'channel': [6, 2, 8, 6],
               'class': [4, 9, 8, 7],
               'load': ['HITProcess', 'XJTUProcess', 'MCC5Process', 'DIRGProcess']}

df = pd.DataFrame(data_select).set_index('name')
parser = argparse.ArgumentParser()

# about data process
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--lr_decay', type=float, default=0.1)

parser.add_argument('--gcug', type=bool, default=True, help='use fast Fourier convolution Unit or not in Global Convolution')
parser.add_argument('--attg', type=bool, default=False, help='use Attention or not in Global Convolution')
parser.add_argument('--attb', type=bool, default=True,  help='use fast Fourier convolution Unit or not in Block')
parser.add_argument('--gcub', type=bool, default=True, help='use fast Fourier convolution Unit or not in Block')
parser.add_argument('--blocker', type=bool, default=True, help='use Gradient Blocking Layer or not in Channel Attention machine')

parser.add_argument('--snr', type=float, default=None)
parser.add_argument('--head', type=int, default=2)
parser.add_argument('--GPU', type=str, default='cuda:2')
parser.add_argument('--log_name', type=str, default="log_name.log")
parser.add_argument('--algorithm', type=str, default='GCA', choices=['GCA', 'softshape', 'PatchTST', 'UniTS', 'DLinear', 'TCN', 'InceptionTime', 'NAT', 'MA1DCNN', 'ConvTran'])
parser.add_argument('--use_data', type=str, default='dirg', choices=['hit', 'xjtu', 'mcc5', 'dirg'])

parser.add_argument('--alpha', type=float, default=1)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--windows', type=int, default=512, help='数据处理部分切片的stride')
parser.add_argument('--length', type=int, default=512, help='数据处理部分的序列长度')

data_name = parser.parse_args().use_data
inf = df.loc[data_name]
parser.add_argument('--log_interval', type=int, default=70)
parser.add_argument('--best_model', type=float, default=1, help='the hightest accuracy for model')
parser.add_argument('--layer', type=list, default=layers, help='MobileNet setting')
parser.add_argument('--path', type=str, default=inf['path'], choices={'XJTU': './data/XJTU/XJTU_Gearbox/', 'MCC5': './data/MCC5_THU/'})
parser.add_argument('--in_channel', type=int, default=inf['channel'], choices={'HIT': 6, 'XJTU': 5, 'MCC5_THU': 8})
parser.add_argument('--class_num', type=int, default=inf['class'], choices={'HIT': 4, 'XJTU': 9, 'MCC5_THU': 8})
parser.add_argument('--load_data', type=str, default=inf['load'])
parser.add_argument('--class_count', type=list, default=None)

parser.add_argument('--lambda_l2', type=float, default=1e-4)

args = parser.parse_args()
