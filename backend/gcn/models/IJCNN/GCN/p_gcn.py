from torch.nn import Parameter

from backend.gcn.models.graph_convolution import GraphConvolution
from backend.gcn.utils import *


class ResnetPGCN(nn.Module):
    def __init__(self, model, num_classes, in_channel=300, out_channels=2048, adjList=None):
        super(ResnetPGCN, self).__init__()

        self.features = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
        )
        self.num_classes = num_classes
        self.pooling = nn.MaxPool2d(14, 14)

        self.fc = nn.Linear(in_features=in_channel, out_features=num_classes)
        torch.nn.init.kaiming_normal_(self.fc.weight.data)

        self.gc1 = GraphConvolution(in_channel, 1024)
        self.gc2 = GraphConvolution(1024, out_channels)
        self.relu = nn.LeakyReLU(0.2)

        self.A = self.generateA(adjList)

    def generateA(self, adjList):
        return Parameter(torch.from_numpy(adjList).float())

    def updateA(self, adjList):
        self.A.data.copy_(torch.from_numpy(adjList).float())

    def forward(self, feature, inp, _adj=None):
        feature = self.features(feature)
        feature = self.pooling(feature)

        feature = feature.view(feature.size(0), -1)

        y1 = self.fc(feature)

        gcn_input = (feature.unsqueeze(2).expand(-1, -1, self.num_classes).transpose(1,
                                                                                     2) * self.fc.weight)

        adj = gen_adj(self.A).detach()
        x = self.gc1(gcn_input, adj)
        x = self.relu(x)
        x = self.gc2(x, adj)

        y2 = x.view((x.size(0), -1))
        return y1 + y2

    def get_config_optim(self, lr=0.1, lrp=0.1):
        return [
            {'params': self.features.parameters(), 'lr': lr * lrp},
            {'params': self.gc1.parameters(), 'lr': lr},
            {'params': self.gc2.parameters(), 'lr': lr},

            {'params': self.fc.parameters(), 'lr': lr}
        ]

    def get_feature_params(self):
        return self.features.parameters()

    def get_gcn_params(self):
        return [
            {'params': self.gc1.parameters()},
            {'params': self.gc2.parameters()}
        ]
