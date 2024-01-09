import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from backend.gcn.models.graph_convolution import GraphConvolution
from backend.gcn.models.salgl import EntropyLoss, LowRankBilinearAttention
from backend.gcn.utils import gen_adjs
from torch.nn import Parameter


class Element_Wise_Layer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(Element_Wise_Layer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(in_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        for i in range(self.in_features):
            self.weight[i].data.uniform_(-stdv, stdv)
        if self.bias is not None:
            for i in range(self.in_features):
                self.bias[i].data.uniform_(-stdv, stdv)

    def forward(self, input):
        x = input * self.weight
        x = torch.sum(x, 2)
        if self.bias is not None:
            x = x + self.bias
        return x

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None)


class ResnetAggSalgl(nn.Module):

    def __init__(self, model, img_size=448, embed_dim=300, feat_dim=2048, att_dim=1024, num_scenes=6,
                 gcn_middle_dim=1024, out_channels=2048, num_classes=80):

        super(ResnetAggSalgl, self).__init__()

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
        self.num_scenes = num_scenes
        self.num_classes = num_classes

        self.total_scene_cnts = [0] * num_scenes
        self.scene_linear = nn.Linear(feat_dim, num_scenes, bias=False)

        self.attention = LowRankBilinearAttention(feat_dim, embed_dim, att_dim)

        self.gc1 = GraphConvolution(feat_dim, gcn_middle_dim)
        self.gc2 = GraphConvolution(gcn_middle_dim, out_channels)
        self.relu = nn.LeakyReLU(0.2)

        self.register_buffer('comatrix', torch.zeros((num_scenes, num_classes, num_classes)))

        self.entropy = EntropyLoss()
        self.max_en = self.entropy(torch.tensor([1 / num_scenes] * num_scenes))

        self.fc = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim),
            nn.Tanh()
        )
        self.classifier = Element_Wise_Layer(num_classes, feat_dim)

    def comatrix2prob(self):

        comat = self.comatrix

        temp = torch.diagonal(comat, dim1=1, dim2=2).unsqueeze(-1)

        comat = comat / (temp + 1e-8)

        for i in range(self.num_classes):
            comat[:, i, i] = 1

        return comat


    def forward(self, x, inp, y=None):
        img_feats = self.features(x)
        img_feats = torch.flatten(img_feats, start_dim=2).transpose(1, 2)

        img_contexts = torch.mean(img_feats, dim=1)

        scene_scores = self.scene_linear(img_contexts)

        if self.training:
            _scene_scores = scene_scores
            _scene_probs = F.softmax(_scene_scores, dim=-1)

            batch_comats = torch.bmm(y.unsqueeze(2), y.unsqueeze(1))

            for i in range(_scene_probs.shape[0]):
                max_scene_id = torch.argmax(_scene_probs[i])

                self.comatrix[max_scene_id] += batch_comats[i]

                self.total_scene_cnts[max_scene_id] += 1

        scene_probs = F.softmax(scene_scores, dim=-1)

        sample_en = self.entropy(scene_probs)

        _scene_probs = torch.mean(scene_probs, dim=0)
        batch_en = self.max_en - self.entropy(_scene_probs)

        label_feats, alphas = self.attention(img_feats, inp)

        comats = self.comatrix2prob()

        indices = torch.argmax(scene_probs, dim=-1)
        comats = torch.index_select(comats, dim=0, index=indices)

        adjs = gen_adjs(comats)

        x = self.gc1(label_feats, adjs)
        x = self.relu(x)
        x = self.gc2(x, adjs)

        output = torch.cat([label_feats, x], dim=-1)
        output = self.fc(output)
        output = self.classifier(output)

        return {
            'output': output,
            'scene_probs': scene_probs,
            'entropy_loss': sample_en + batch_en,
            'comat': comats
        }

    def get_config_optim(self, lr=0.1, lrp=0.1):
        return [
            {'params': self.features.parameters(), 'lr': lr * lrp},
            {'params': self.gc1.parameters(), 'lr': lr},
            {'params': self.gc2.parameters(), 'lr': lr},
            {'params': self.attention.parameters(), 'lr': lr},
            {'params': self.fc.parameters(), 'lr': lr},

            {'params': self.classifier.parameters(), 'lr': lr},

            {'params': self.scene_linear.parameters(), 'lr': lr}
        ]
