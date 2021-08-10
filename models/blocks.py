import torch

from torch import nn
from torch.nn import Sequential as Seq, Linear as Lin, Conv1d

def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
    """
    activation layer
    :param act:
    :param inplace:
    :param neg_slope:
    :param n_prelu:
    :return:
    """

    act = act.lower()
    if act == 'relu':
        layer = nn.ReLU(inplace)
    elif act == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act)
    return layer

        
# Now, let's implement a sharedMLP layer. It is implmented by using Conv1d with kernel size equals to 1. 
class Conv1dLayer(Seq):
    def __init__(self, channels, act='relu', norm=True, bias=True):
        m = []
        for i in range(1, len(channels)):
            m.append(Conv1d(channels[i - 1], channels[i], 1, bias=bias))
            if norm:
                m.append(nn.BatchNorm1d(channels[i]))
            if act:
                m.append(act_layer(act))
        super(Conv1dLayer, self).__init__(*m)


class MLP(Seq):
    """
    Given input with shape [B, C_in]
    return output with shape [B, C_out] 
    """

    def __init__(self, channels, act='relu', norm=True, bias=True, dropout=0.5):
        # todo:
        m = []
        for i in range(1, len(channels)):
            m.append(Lin(channels[i - 1], channels[i], bias=bias))
            if norm:
                m.append(nn.BatchNorm1d(channels[i]))
            if act:
                m.append(act_layer(act))
            if dropout > 0:
                m.append(nn.Dropout(dropout))
        super(MLP, self).__init__(*m)

