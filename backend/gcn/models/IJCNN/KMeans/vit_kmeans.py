import copy
import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Parameter

from backend.gcn.kmeans_torch import kmeans
from backend.gcn.models.graph_convolution import GraphConvolution
from backend.gcn.models.salgl import LowRankBilinearAttention
from backend.gcn.utils import gen_adjs


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None, maxH=30, maxW=30):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

        self.maxH = maxH
        self.maxW = maxW
        pe = self._gen_pos_buffer()
        self.register_buffer('pe', pe)

    def _gen_pos_buffer(self):
        _eyes = torch.ones((1, self.maxH, self.maxW))
        y_embed = _eyes.cumsum(1, dtype=torch.float32)
        x_embed = _eyes.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

    def forward(self, input: Tensor):
        x = input
        return self.pe.repeat((x.size(0), 1, 1, 1))


def build_position_encoding(hidden_dim, arch, position_embedding, img_size):
    N_steps = hidden_dim // 2

    if arch in ['CvT_w24'] or 'vit' in arch:
        downsample_ratio = 16
    else:
        downsample_ratio = 32

    if position_embedding in ('v2', 'sine'):

        assert img_size % 32 == 0, "args.img_size ({}) % 32 != 0".format(img_size)
        position_embedding = PositionEmbeddingSine(N_steps, normalize=True, maxH=img_size // downsample_ratio,
                                                   maxW=img_size // downsample_ratio)
    else:
        raise ValueError(f"not supported {position_embedding}")

    return position_embedding


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU(inplace=True)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask=None,
                     src_key_padding_mask=None,
                     pos=None):
        q = k = self.with_pos_embed(src, pos)
        src2, corr = self.self_attn(q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)

        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, norm=None):
        super().__init__()

        self.layers = nn.ModuleList([copy.deepcopy(TransformerEncoderLayer(d_model, nhead)) for _ in range(num_layers)])
        self.norm = norm

    def forward(self, src,
                mask=None,
                src_key_padding_mask=None,
                pos=None):
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask, pos=pos)
        if self.norm is not None:
            output = self.norm(output)
        return output


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


class VitKMeans(nn.Module):

    def __init__(self, model, img_size=448, embed_dim=300, feat_dim=2048, att_dim=1024, num_scenes=6,
                 n_head=8, num_layers=1, gcn_middle_dim=1024,
                 out_channels=2048, num_classes=80, comat_smooth=False):

        super(VitKMeans, self).__init__()

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
        self.comat_smooth = comat_smooth

        self.centers = torch.Tensor(num_scenes, feat_dim)

        torch.nn.init.kaiming_uniform_(self.centers, a=math.sqrt(5))

        self.total_scene_cnts = [0] * num_scenes

        self.transformer = TransformerEncoder(d_model=feat_dim, nhead=n_head, num_layers=num_layers)
        self.position_embedding = build_position_encoding(feat_dim, "resnet101", "sine",
                                                          img_size=img_size)

        self.attention = LowRankBilinearAttention(feat_dim, embed_dim, att_dim)

        self.gc1 = GraphConvolution(feat_dim, gcn_middle_dim)
        self.gc2 = GraphConvolution(gcn_middle_dim, out_channels)
        self.relu = nn.LeakyReLU(0.2)
        self.register_buffer('comatrix', torch.zeros((num_scenes, num_classes, num_classes)))

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

        pos = self.position_embedding(x)
        pos = torch.flatten(pos, 2).transpose(1, 2)

        img_feats = self.transformer(img_feats, pos=pos)

        img_contexts = torch.mean(img_feats, dim=1)

        scene_ids_x, self.centers = kmeans(img_contexts, num_clusters=self.num_scenes, initial_state=self.centers,
                                           scene_cnts=self.total_scene_cnts)

        if self.training:

            batch_comats = torch.bmm(y.unsqueeze(2), y.unsqueeze(1))

            for i in range(len(scene_ids_x)):
                max_scene_id = scene_ids_x[i]

                self.comatrix[max_scene_id] += batch_comats[i]

                self.total_scene_cnts[max_scene_id] += 1

        label_feats, alphas = self.attention(img_feats, inp)

        comats = self.comatrix2prob()

        comats = torch.index_select(comats, dim=0, index=scene_ids_x)

        adjs = gen_adjs(comats)

        x = self.gc1(label_feats, adjs)
        x = self.relu(x)
        x = self.gc2(x, adjs)

        output = torch.cat([label_feats, x], dim=-1)
        output = self.fc(output)
        output = self.classifier(output)

        return {
            'output': output,
            'comat': comats
        }

    def get_config_optim(self, lr=0.1, lrp=0.1):
        return [
            {'params': self.features.parameters(), 'lr': lr * lrp},
            {'params': self.gc1.parameters(), 'lr': lr},
            {'params': self.gc2.parameters(), 'lr': lr},
            {'params': self.transformer.parameters(), 'lr': lr},
            {'params': self.attention.parameters(), 'lr': lr},
            {'params': self.fc.parameters(), 'lr': lr},

            {'params': self.classifier.parameters(), 'lr': lr}
        ]
