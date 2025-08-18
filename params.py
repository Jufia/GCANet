import argparse
import pandas as pd

data_select = {'name': ['hit', 'xjtu', 'mcc5', 'dirg'],
               'path': ['./data/hit/hit/', 
                        './data/XJTU/XJTU/', 
                        './data/MCC5_THU/mcc5/',
                        './data/dirg/'],
               'channel': [6, 2, 8, 6],
               'class': [4, 9, 8, 7],
               'load': ['HITProcess', 'XJTUProcess', 'MCC5Process', 'DIRGProcess']}

df = pd.DataFrame(data_select).set_index('name')
parser = argparse.ArgumentParser()

# about data process
parser.add_argument('--random_state', type=int, default=3470)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--lr_decay', type=float, default=0.1)

parser.add_argument('--gcug', type=str, default='gcu', choices=['gcu', 'fft', 'none'], help='use fast Fourier convolution Unit or not in Global Convolution')
parser.add_argument('--att', type=str, default='agca', choices=['se', 'agca', 'none'], help='use fast Fourier convolution Unit or not in Block')
parser.add_argument('--gcub', type=str, default='gcu', choices=['gcu', 'fft', 'none'], help='use fast Fourier convolution Unit or not in Block')
parser.add_argument('--blocker', type=bool, default=True, help='use Gradient Blocking Layer or not in Channel Attention machine')

parser.add_argument('--snr', type=float, default=None)
parser.add_argument('--head', type=int, default=2)
parser.add_argument('--GPU', type=str, default='cuda:1')
parser.add_argument('--log_name', type=str, default="test.log")
parser.add_argument('--algorithm', type=str, default='GCA')
parser.add_argument('--use_data', type=str, default='hit', choices=['hit', 'xjtu', 'mcc5', 'dirg'])
parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd', 'radam'])

parser.add_argument('--alpha', type=float, default=1)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--length', type=int, default=512, help='数据处理部分的序列长度')

parser.add_argument('--lambda_l2', type=float, default=1e-4)

data_name = parser.parse_args().use_data
info = df.loc[data_name]
parser.add_argument('--log_interval', type=int, default=70)
parser.add_argument('--best_model', type=float, default=0, help='the hightest accuracy for model')
parser.add_argument('--path', type=str, default=info['path'], choices={'XJTU': './data/XJTU/XJTU_Gearbox/', 'MCC5': './data/MCC5_THU/'})
parser.add_argument('--in_channel', type=int, default=info['channel'], choices={'HIT': 6, 'XJTU': 5, 'MCC5_THU': 8})
parser.add_argument('--class_num', type=int, default=info['class'], choices={'HIT': 4, 'XJTU': 9, 'MCC5_THU': 8})
parser.add_argument('--load_data', type=str, default=info['load'])
parser.add_argument('--class_count', type=list, default=None)

args = parser.parse_args()
