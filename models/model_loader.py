import torch
from torch import nn
from models.crnn import CRNN
from collections import OrderedDict

def load_weights(target, source_state):
    new_dict = OrderedDict()
    for k, v in target.state_dict().items():
        if k in source_state and v.size() == source_state[k].size():
            new_dict[k] = source_state[k]
        else:
            new_dict[k] = v
    target.load_state_dict(new_dict)

def load_model(num_classes, seq_proj=[0, 0], backend='resnet18', snapshot=None, cuda=True):
    net = CRNN(num_classes=num_classes, seq_proj=seq_proj, backend=backend)
    net = nn.DataParallel(net)
    if snapshot is not None:
        load_weights(net, torch.load(snapshot))
    if cuda:
        net = net.cuda()
    return net
