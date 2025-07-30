from .GCAL import GCANet

# ConvTran
from baseline.PatchTST.PatchTST import patchtstModel
from baseline.SoftShape.SoftShape import SoftShapeNet

from baseline.TimesNet.timesnet import Model as TimesNet

from baseline.ModernTCN.ModernTCN import Model as ModernTCNd
from baseline.TimeMixer.TimeMixer import Model as TimeMixer

from baseline.DLinear.DLinear import dlinear
from baseline.LightTS.LightTS import Model as LightTS

from baseline.Inception.inception import InceptionTime
from baseline.TCN.tcn import TCN


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