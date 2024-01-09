import torch
import torch.nn as nn

import torch.nn.functional as F
from torch.nn import Parameter

from backend.gcn.models.graph_convolution import GraphConvolution
from backend.gcn.models.salgl import EntropyLoss, LowRankBilinearAttention
from backend.gcn.utils import gen_adjs, gen_adj
import math
from torch import Tensor
from typing import Optional

import copy


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


# 位置编码策略
def build_position_encoding(hidden_dim, arch, position_embedding, img_size):
    N_steps = hidden_dim // 2

    if arch in ['CvT_w24'] or 'vit' in arch:
        downsample_ratio = 16
    else:
        downsample_ratio = 32

    if position_embedding in ('v2', 'sine'):
        # TODO find a better way of exposing other arguments
        assert img_size % 32 == 0, "args.img_size ({}) % 32 != 0".format(img_size)
        position_embedding = PositionEmbeddingSine(N_steps, normalize=True, maxH=img_size // downsample_ratio,
                                                   maxW=img_size // downsample_ratio)
    else:
        raise ValueError(f"not supported {position_embedding}")

    return position_embedding


# 定义单个编码层
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        # 自注意力层
        # batch_first的含义：batch是否在第一维，如果True，则输入的维度是(batch, seq, feature)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        # 前馈模块
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)  # 一个是在多头注意力模块后进行norm
        self.norm2 = nn.LayerNorm(d_model)  # 另一个是在feedforward后进行norm
        self.dropout1 = nn.Dropout(dropout)  # 两个模块都进行dropout
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU(inplace=True)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask=None,
                     src_key_padding_mask=None,
                     pos=None):
        # q和k的输入是相同的，加入了位置编码，挖掘的是自注意力机制
        q = k = self.with_pos_embed(src, pos)
        src2, corr = self.self_attn(q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        # 残差连接并进行norm
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        # 经过前馈网络
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


# 定义整个编码模块
class TransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, norm=None):
        super().__init__()
        # 连续堆叠num_layers个编码块
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


# 定义SAL-GL模型
# 对于每个网络层，都进行权重初始化
#  1. self.features 使用预训练权重
#  2. nn.Linear 已经进行了权重初始化-->恺明初始化
#  3. 自定义的Parameter也已经进行了权重初始化
class FullSALGL(nn.Module):
    # 传进来一个resnet101，截取其前一部分作为主干网络
    # Todo: 比较重要的超参数
    #  1. 场景个数 num_scenes=6
    #  2. transformer相关：n_head=4、num_layers=1
    #  3. 低秩双线性注意力中联合嵌入空间的维度：att_dim= 1024(300维和2048计算相关性)
    #  4. 图卷积神经网络部分，中间层的维度：gcn_middle_dim=1024
    #                     最后输出层的维度out_channels要和feat_dim相等，以进行concat操作
    #  5. 是否进行相关性矩阵平滑：comat_smooth=False
    #  
    def __init__(self, model, img_size=448, embed_dim=300, feat_dim=2048, att_dim=1024, num_scenes=6,
                 n_head=8, num_layers=1, gcn_middle_dim=1024,
                 out_channels=2048, num_classes=80, comat_smooth=False):

        super(FullSALGL, self).__init__()
        # cnn主干网络，提取特征
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

        # 计算场景的线性分类器
        self.scene_linear = nn.Linear(feat_dim, num_scenes, bias=False)

        # 引入transformer结构
        self.transformer = TransformerEncoder(d_model=feat_dim, nhead=n_head, num_layers=num_layers)
        self.position_embedding = build_position_encoding(feat_dim, "resnet101", "sine",
                                                          img_size=img_size)  # 这里的img_size应该是图像特征的img_size
        # 低秩双线性注意力
        self.attention = LowRankBilinearAttention(feat_dim, embed_dim, att_dim)

        # 图卷积网络部分
        self.gc1 = GraphConvolution(feat_dim, gcn_middle_dim)
        self.gc2 = GraphConvolution(gcn_middle_dim, out_channels)
        self.relu = nn.LeakyReLU(0.2)
        self.register_buffer('comatrix', torch.zeros((num_scenes, num_classes, num_classes)))

        self.entropy = EntropyLoss()
        # 计算最大熵，预测向量呈均匀分布时的熵
        self.max_en = self.entropy(torch.tensor([1 / num_scenes] * num_scenes))

        # embed_dim是标签嵌入的维度，300维？
        # embed_dim = 300
        self.fc = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim),
            nn.Tanh()
        )
        self.classifier = Element_Wise_Layer(num_classes, feat_dim)

    # 将(对称的)共现矩阵转换成(非对称的)概率矩阵
    def comatrix2prob(self):
        # comat: [num_scenes,nc,nc]
        # self.comatrix在训练时更新
        # Todo: 这里先不进行转置，pij仍然表示标签i出现的条件下标签j出现的
        # comat = torch.transpose(self.comatrix, dim0=1, dim1=2)
        comat = self.comatrix
        # 计算每个场景中每个标签的出现次数，作为分母
        temp = torch.diagonal(comat, dim1=1, dim2=2).unsqueeze(-1)
        # 标签共现矩阵除以每个标签的出现次数，即为概率矩阵
        comat = comat / (temp + 1e-8)
        # 主对角线部分设置为1
        for i in range(self.num_classes):
            comat[:, i, i] = 1
        # 引入自连接
        return comat

    """
    x: 图像特征
    y: 图像标签集合
    inp: 标签嵌入
    """

    def forward(self, x, inp, y=None):
        img_feats = self.features(x)
        # 将后两个维度H和W进行合并，然后放到第2维上面
        # 此时的维度是(batch_size, channel, h * w)
        img_feats = torch.flatten(img_feats, start_dim=2).transpose(1, 2)

        # 计算位置编码
        pos = self.position_embedding(x)
        pos = torch.flatten(pos, 2).transpose(1, 2)
        # 将关于位置的特征向量序列和位置编码输入到transformer中，自注意力机制输出和输入的img_feats形状相同
        img_feats = self.transformer(img_feats, pos=pos)
        # 在每个位置上求平均值，得到(batch_size,feat_dim)，然后输入到场景分类器中求场景分布
        img_contexts = torch.mean(img_feats, dim=1)
        # scene_cnts是该批次样本中每个场景的分类样本数
        scene_cnts = [0] * self.num_scenes
        scene_scores = self.scene_linear(img_contexts)
        if self.training:
            _scene_scores = scene_scores
            _scene_probs = F.softmax(_scene_scores, dim=-1)
            # y的维度是(B, num_labels)
            # batch_comats的维度是(B, num_labels, num_labels) 是0-1矩阵，如果为1，说明对应标签对同时出现了
            batch_comats = torch.bmm(y.unsqueeze(2), y.unsqueeze(1))
            # _scene_probs的维度是(B, num_scenes)
            for i in range(_scene_probs.shape[0]):
                # 采用各个场景的标签共现矩阵加权还是说只取最大概率场景的标签共现矩阵
                if self.comat_smooth:
                    prob = _scene_probs[i].unsqueeze(-1).unsqueeze(-1)  # [num_scenes, 1, 1]
                    comat = batch_comats[i].unsqueeze(0)  # [1, num_labels, num_labels]
                    # 当前comat对每个场景都有贡献，贡献值即为scene的预测概率值
                    self.comatrix += prob * comat  # [num_scenes, num_labels, num_labels]
                    # Todo: 当为矩阵更新方式为soft时，scene_cnts的统计变更
                else:
                    max_scene_id = torch.argmax(_scene_probs[i])
                    # comats只加到对应的最大概率场景的标签共现矩阵上
                    self.comatrix[max_scene_id] += batch_comats[i]
                    # 更新scene_cnts
                    scene_cnts[max_scene_id] += 1
        # 计算场景的概率
        scene_probs = F.softmax(scene_scores, dim=-1)
        # 计算场景概率的熵损失
        sample_en = self.entropy(scene_probs)
        # 计算该批次场景预测向量的平均值
        _scene_probs = torch.mean(scene_probs, dim=0)
        # 计算和最大熵的差距，应该使这种差距尽可能小，使场景的预测更加多样化，缓解winner-take-all的问题
        # 这个是批次平均概率向量的熵，而前面的是概率向量熵的平均值，两者代表的是不同的含义
        batch_en = self.max_en - self.entropy(_scene_probs)

        # attention计算每个标签的视觉表示
        # Todo: Transformer挖掘空间特征，复习Transformer
        label_feats, alphas = self.attention(img_feats, inp)

        # 根据场景预测向量选择合适的标签共现矩阵
        if not self.comat_smooth:
            comats = self.comatrix2prob()
            # 取出每张图像对应的概率最大的场景索引
            indices = torch.argmax(scene_probs, dim=-1)
            # 取出对应的标签概率矩阵
            comats = torch.index_select(comats, dim=0, index=indices)
        else:  # 否则的话，进行一下加权
            _scene_probs = scene_probs.unsqueeze(-1).unsqueeze(-1)  # [bs,num_scenes,1,1]
            comats = self.comatrix2prob().unsqueeze(0)  # [1,num_scenes,nc,nc]
            comats = _scene_probs * comats  # [bs,num_scenes,nc,nc]
            # 根据场景预测概率向量对每个场景的标签概率矩阵进行加权，即soft策略
            comats = torch.sum(comats, dim=1)  # [bs,nc,nc]

        # 输入标签的视觉特征和共现矩阵
        # label_feats的维度是[batch_size, num_labels, att_dim]
        # comats的维度是[batch_size, num_labels, num_labels]
        # 与传统的GCN相比多了一个batch维度
        # 1. 先对comats进行adj的转换
        adjs = gen_adjs(comats)
        # 每张图片采用不同的邻接矩阵
        x = self.gc1(label_feats, adjs)
        x = self.relu(x)
        x = self.gc2(x, adjs)  # [batch_size, num_classes, feat_dim]

        output = torch.cat([label_feats, x], dim=-1)
        output = self.fc(output)
        output = self.classifier(output)
        # Todo: 求点积，得到预测向量
        # 返回前向传播的结果
        return {
            'output': output,
            'scene_probs': scene_probs,
            'entropy_loss': sample_en + batch_en,
            'comat': comats,
            'scene_cnts': scene_cnts
        }

    # Todo: 获取需要优化的参数
    def get_config_optim(self, lr=0.1, lrp=0.1):
        return [
            {'params': self.features.parameters(), 'lr': lr * lrp},
            {'params': self.gc1.parameters(), 'lr': lr},
            {'params': self.gc2.parameters(), 'lr': lr},
            {'params': self.transformer.parameters(), 'lr': lr},
            {'params': self.attention.parameters(), 'lr': lr},
            {'params': self.fc.parameters(), 'lr': lr},
            # Todo: Debug查看一下分类器参数
            {'params': self.classifier.parameters(), 'lr': lr},
            # Todo: 以下是不能聚合的参数
            {'params': self.scene_linear.parameters(), 'lr': lr}
        ]
