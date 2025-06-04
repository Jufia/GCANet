# from .LSTM import LSTM
# from .Transformer import Transformer
# from .TCN import TCN
# from .Inception import InceptionTime
# from .NAT import NAT
# from .MA1DCNN import MA1DCNN
# from .ConTran import ConvTran
# from .TST import TSTransformerEncoderClassiregressor
from .GCAL import GCALNet
from baseline.SoftShape.SoftShape import SoftShapeNet
from baseline.PatchTST.PatchTST import patchtstModel


model_dic = {
                'GCA': GCALNet,
                'softshape': SoftShapeNet,
                'PatchTST': patchtstModel,
             }