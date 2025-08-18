from .GCAL import GCANet

# ConvTran
from baseline.ConvTran.model import ConvTran
from baseline.PatchTST.PatchTST import patchtstModel
from baseline.SoftShape.SoftShape import SoftShapeNet
from baseline.UniTS.UniTS import units

from baseline.TimesNet.timesnet import Model as TimesNet

from baseline.ModernTCN.ModernTCN import Model as ModernTCNd
from baseline.TimeMixer.TimeMixer import Model as TimeMixer

from baseline.DLinear.DLinear import dlinear
from baseline.LightTS.LightTS import Model as LightTS

from baseline.Inception.inception import InceptionTime
from baseline.TCN.tcn import TCN
from baseline.biLSTM.biLSTM import biLSTM


model_dic = {
                'GCA': GCANet,
                'ConvTran': ConvTran,
                'PatchTST': patchtstModel,
                'softshape': SoftShapeNet,
                'UniTS': units,
                'TimesNet': TimesNet,
                'ModernTCN': ModernTCNd,
                'TimeMixer': TimeMixer,
                'DLinear': dlinear,
                'LightTS': LightTS,
                'InceptionTime': InceptionTime,
                'TCN': TCN,
                'biLSTM': biLSTM,
             }