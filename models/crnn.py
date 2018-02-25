import torch
import torch.nn as nn
from torch.autograd import Variable

import torchvision.models as models

class CRNN(nn.Module):
    def __init__(self,
                 num_classes=10,
                 backend='resnet18',
                 rnn_hidden_size=128,
                 rnn_num_layers=2,
                 rnn_dropout=False,
                 seq_proj=[0, 0]):
        super().__init__()

        self.feature_extractor = getattr(models, backend)(pretrained=True)
        self.cnn = nn.Sequential(
            self.feature_extractor.conv1,
            self.feature_extractor.bn1,
            self.feature_extractor.relu,
            self.feature_extractor.maxpool,
            self.feature_extractor.layer1,
            self.feature_extractor.layer2,
            self.feature_extractor.layer3,
            self.feature_extractor.layer4
        )

        self.fully_conv = seq_proj[0] == 0
        if not self.fully_conv:
            self.proj = nn.Conv2d(seq_proj[0], seq_proj[1], kernel_size=1)

        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_num_layers = rnn_num_layers
        self.rnn = nn.GRU(self.get_block_size(self.cnn),
                          rnn_hidden_size, rnn_num_layers,
                          batch_first=False,
                          dropout=rnn_dropout, bidirectional=True)
        self.linear = nn.Linear(rnn_hidden_size * 2, num_classes + 1)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        hidden = self.init_hidden(x.size(0), next(self.parameters()).is_cuda)
        features = self.cnn(x)
        features = self.features_to_sequence(features)
        seq, hidden = self.rnn(features, hidden)
        seq = self.linear(seq)
        if not self.training:
            seq = self.softmax(seq)
        return seq

    def init_hidden(self, batch_size, gpu=False):
        h0 = Variable(torch.zeros( self.rnn_num_layers * 2,
                                   batch_size,
                                   self.rnn_hidden_size))
        if gpu:
            h0 = h0.cuda()
        return h0

    def features_to_sequence(self, features):
        b, c, h, w = features.size()
        assert h == 1, "the height of out must be 1"
        if not self.fully_conv:
            features = features.permute(0, 3, 2, 1)
            features = self.proj(features)
            features = features.permute(1, 0, 2, 3)
        else:
            features = features.permute(3, 0, 2, 1)
        features = features.squeeze(2)
        return features

    def get_block_size(self, layer):
        return layer[-1][-1].bn2.weight.size()[0]
