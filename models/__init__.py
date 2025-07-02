# from .LSTM import LSTM
# from .Transformer import Transformer
# from .TCN import TCN
# from .Inception import InceptionTime
# from .NAT import NAT
# from .MA1DCNN import MA1DCNN
# from .ConTran import ConvTran
# from .TST import TSTransformerEncoderClassiregressor
from .GCAL import GCANet
from baseline.SoftShape.SoftShape import SoftShapeNet
from baseline.PatchTST.PatchTST import patchtstModel
from baseline.UniTS.UniTS import units
from baseline.DLinear.DLinear import dlinear


model_dic = {
                'GCA': GCANet,
                'softshape': SoftShapeNet,
                'PatchTST': patchtstModel,
                'UniTS': units,
                'DLinear': dlinear,
             }