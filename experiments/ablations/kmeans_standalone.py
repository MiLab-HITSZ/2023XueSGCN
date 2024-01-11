import csv
import json
import math
import os
import os.path
import pickle
import random

import numpy as np
import torch
import torch.nn
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset


class AveragePrecisionMeter(object):

    def __init__(self, difficult_examples=False):
        super(AveragePrecisionMeter, self).__init__()
        self.reset()
        self.difficult_examples = difficult_examples

    def reset(self):
        self.scores = torch.FloatTensor(torch.FloatStorage())
        self.targets = torch.LongTensor(torch.LongStorage())

    def add(self, output, target):

        if not torch.is_tensor(output):
            output = torch.from_numpy(output)
        if not torch.is_tensor(target):
            target = torch.from_numpy(target)

        if output.dim() == 1:
            output = output.view(-1, 1)
        else:
            assert output.dim() == 2, \
                'wrong output size (should be 1D or 2D with one column \
                per class)'
        if target.dim() == 1:
            target = target.view(-1, 1)
        else:
            assert target.dim() == 2, \
                'wrong target size (should be 1D or 2D with one column \
                per class)'
        if self.scores.numel() > 0:
            assert target.size(1) == self.targets.size(1), \
                'dimensions for output should match previously added examples.'

        if self.scores.storage().size() < self.scores.numel() + output.numel():
            new_size = math.ceil(self.scores.storage().size() * 1.5)
            self.scores.storage().resize_(int(new_size + output.numel()))
            self.targets.storage().resize_(int(new_size + output.numel()))

        offset = self.scores.size(0) if self.scores.dim() > 0 else 0
        self.scores.resize_(offset + output.size(0), output.size(1))
        self.targets.resize_(offset + target.size(0), target.size(1))
        self.scores.narrow(0, offset, output.size(0)).copy_(output)
        self.targets.narrow(0, offset, target.size(0)).copy_(target)

    def value(self):

        if self.scores.numel() == 0:
            return 0

        ap = torch.full((self.scores.size(1),), -1.)

        non_zero_labels = 0
        non_zero_ap_sum = 0
        for k in range(self.scores.size(1)):
            targets = self.targets[:, k]

            if targets.sum() == 0:
                continue
            non_zero_labels += 1

            scores = self.scores[:, k]

            ap[k] = AveragePrecisionMeter.average_precision(scores, targets, self.difficult_examples)
            non_zero_ap_sum += ap[k]

        mAP = non_zero_ap_sum / non_zero_labels

        return mAP, ap.tolist()

    @staticmethod
    def average_precision(output, target, difficult_examples=False):

        sorted, indices = torch.sort(output, dim=0, descending=True)

        pos_count = 0.
        total_count = 0.
        precision_at_i = 0.

        for i in indices:
            label = target[i]
            if difficult_examples and label == 0:
                continue

            if label == 1:
                pos_count += 1

            total_count += 1
            if label == 1:
                precision_at_i += pos_count / total_count

        if pos_count != 0:
            precision_at_i /= pos_count

        return precision_at_i

    def overall(self):
        if self.scores.numel() == 0:
            return 0
        scores = self.scores.cpu().numpy()
        targets = self.targets.cpu().numpy()
        targets[targets == -1] = 0
        return self.evaluation(scores, targets)

    def overall_topk(self, k):
        targets = self.targets.cpu().numpy()
        targets[targets == -1] = 0
        n, c = self.scores.size()
        scores = np.zeros((n, c)) - 1
        index = self.scores.topk(k, 1, True, True)[1].cpu().numpy()
        tmp = self.scores.cpu().numpy()
        for i in range(n):
            for ind in index[i]:
                scores[i, ind] = 1 if tmp[i, ind] >= 0 else -1
        return self.evaluation(scores, targets)

    def evaluation(self, scores_, targets_):
        n, n_class = scores_.shape
        Nc, Np, Ng = np.zeros(n_class), np.zeros(n_class), np.zeros(n_class)
        for k in range(n_class):
            scores = scores_[:, k]
            targets = targets_[:, k]
            targets[targets == -1] = 0

            Ng[k] = np.sum(targets == 1)

            Np[k] = np.sum(scores >= 0)

            Nc[k] = np.sum(targets * (scores >= 0))

        if np.sum(Np) == 0:
            OP = -1
            OR = -1
            OF1 = -1
        else:
            OP = np.sum(Nc) / np.sum(Np)
            OR = np.sum(Nc) / np.sum(Ng)
            OF1 = (2 * OP * OR) / (OP + OR)

        CP_SUM = 0
        CP_CNT = 0
        CR_SUM = 0
        CR_CNT = 0
        CP = -1
        CR = -1
        CF1 = -1

        for i in range(n_class):
            if Np[i] != 0:
                CP_CNT += 1
                CP_SUM += Nc[i] / Np[i]
            if Ng[i] != 0:
                CR_CNT += 1
                CR_SUM += Nc[i] / Ng[i]
        if CP_CNT != 0:
            CP = CP_SUM / CP_CNT
        if CR_CNT != 0:
            CR = CR_SUM / CR_CNT
        if CP != -1 and CR != -1:
            CF1 = (2 * CP * CR) / (CP + CR)
        return OP, OR, OF1, CP, CR, CF1


class AsymmetricLossOptimized(nn.Module):

    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False):
        super(AsymmetricLossOptimized, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

    def forward(self, x, y):

        self.targets = y
        self.anti_targets = 1 - y

        self.xs_pos = x
        self.xs_neg = 1.0 - self.xs_pos

        if self.clip is not None and self.clip > 0:
            self.xs_neg.add_(self.clip).clamp_(max=1)

        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))

        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            self.xs_pos = self.xs_pos * self.targets
            self.xs_neg = self.xs_neg * self.anti_targets
            self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
                                          self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            self.loss *= self.asymmetric_w

        return -self.loss.sum()


class COCO(Dataset):
    def __init__(self, images_dir, config_dir, transforms=None, inp_name=None):
        self.images_dir = images_dir
        self.config_dir = config_dir
        self.transforms = transforms
        self.img_list = []
        self.cat2idx = None
        self.get_anno()

        self.num_classes = len(self.cat2idx)
        self.inp = None
        if inp_name is not None:
            inp_file = os.path.join(self.config_dir, inp_name)
            with open(inp_file, 'rb') as f:
                self.inp = pickle.load(f)
            self.inp_name = inp_name

    def get_anno(self):
        list_path = os.path.join(self.images_dir, 'anno.json')
        self.img_list = json.load(open(list_path, 'r'))

        category_path = os.path.join(self.config_dir, 'category.json')
        self.cat2idx = json.load(open(category_path, 'r'))

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        item = self.img_list[index]
        img, target = self.get(item)

        if self.inp is not None:
            return (img, self.inp), target
        else:
            return img, target

    def get(self, item):
        filename = item['file_name']
        labels = sorted(item['labels'])

        img = Image.open(os.path.join(self.images_dir, filename)).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)

        target = np.zeros(self.num_classes, np.float32)
        target[labels] = 1
        return img, target


class Warp(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = int(size)
        self.interpolation = interpolation

    def __call__(self, img):
        return img.resize((self.size, self.size), self.interpolation)

    def __str__(self):
        return self.__class__.__name__ + ' (size={size}, interpolation={interpolation})'.format(size=self.size,
                                                                                                interpolation=self.interpolation)


class MultiScaleCrop(object):

    def __init__(self, input_size, scales=None, max_distort=1, fix_crop=True, more_fix_crop=True):
        self.scales = scales if scales is not None else [1, 875, .75, .66]
        self.max_distort = max_distort
        self.fix_crop = fix_crop
        self.more_fix_crop = more_fix_crop
        self.input_size = input_size if not isinstance(input_size, int) else [input_size, input_size]
        self.interpolation = Image.BILINEAR

    def __call__(self, img):
        im_size = img.size
        crop_w, crop_h, offset_w, offset_h = self._sample_crop_size(im_size)
        crop_img_group = img.crop((offset_w, offset_h, offset_w + crop_w, offset_h + crop_h))
        ret_img_group = crop_img_group.resize((self.input_size[0], self.input_size[1]), self.interpolation)
        return ret_img_group

    def _sample_crop_size(self, im_size):
        image_w, image_h = im_size[0], im_size[1]

        base_size = min(image_w, image_h)
        crop_sizes = [int(base_size * x) for x in self.scales]
        crop_h = [self.input_size[1] if abs(x - self.input_size[1]) < 3 else x for x in crop_sizes]
        crop_w = [self.input_size[0] if abs(x - self.input_size[0]) < 3 else x for x in crop_sizes]

        pairs = []
        for i, h in enumerate(crop_h):
            for j, w in enumerate(crop_w):
                if abs(i - j) <= self.max_distort:
                    pairs.append((w, h))

        crop_pair = random.choice(pairs)
        if not self.fix_crop:
            w_offset = random.randint(0, image_w - crop_pair[0])
            h_offset = random.randint(0, image_h - crop_pair[1])
        else:
            w_offset, h_offset = self._sample_fix_offset(image_w, image_h, crop_pair[0], crop_pair[1])

        return crop_pair[0], crop_pair[1], w_offset, h_offset

    def _sample_fix_offset(self, image_w, image_h, crop_w, crop_h):
        offsets = self.fill_fix_offset(self.more_fix_crop, image_w, image_h, crop_w, crop_h)
        return random.choice(offsets)

    @staticmethod
    def fill_fix_offset(more_fix_crop, image_w, image_h, crop_w, crop_h):
        w_step = (image_w - crop_w) // 4
        h_step = (image_h - crop_h) // 4

        ret = list()
        ret.append((0, 0))
        ret.append((4 * w_step, 0))
        ret.append((0, 4 * h_step))
        ret.append((4 * w_step, 4 * h_step))
        ret.append((2 * w_step, 2 * h_step))

        if more_fix_crop:
            ret.append((0, 2 * h_step))
            ret.append((4 * w_step, 2 * h_step))
            ret.append((2 * w_step, 4 * h_step))
            ret.append((2 * w_step, 0 * h_step))

            ret.append((1 * w_step, 1 * h_step))
            ret.append((3 * w_step, 1 * h_step))
            ret.append((1 * w_step, 3 * h_step))
            ret.append((3 * w_step, 3 * h_step))

        return ret

    def __str__(self):
        return self.__class__.__name__


def gcn_train_transforms(resize_scale, crop_scale):
    return transforms.Compose([

        transforms.Resize((resize_scale, resize_scale)),
        MultiScaleCrop(crop_scale, scales=(1.0, 0.875, 0.75, 0.66, 0.5), max_distort=2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def gcn_valid_transforms(resize_scale, crop_scale):
    return transforms.Compose([
        Warp(crop_scale),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def train_transforms(resize_scale, crop_scale, is_gcn=False):
    if is_gcn:
        return gcn_train_transforms(resize_scale, crop_scale)
    return transforms.Compose([

        transforms.Resize(resize_scale),

        transforms.RandomResizedCrop(crop_scale),

        transforms.RandomHorizontalFlip(),

        transforms.ToTensor(),

        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def valid_transforms(resize_scale, crop_scale, is_gcn=False):
    if is_gcn:
        return gcn_valid_transforms(resize_scale, crop_scale)
    return transforms.Compose([
        transforms.Resize(resize_scale),

        transforms.CenterCrop(crop_scale),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


class DatasetLoader(object):

    def __init__(self, category_dir, train_path, valid_path, inp_name=None):
        super(DatasetLoader, self).__init__()
        self.category_dir = category_dir
        self.train_path = train_path
        self.valid_path = valid_path
        self.inp_name = inp_name
        self.is_gcn = inp_name is not None

    def get_loaders(self, batch_size, resize_scale=512, crop_scale=448):
        train_dataset = COCO(images_dir=self.train_path,
                             config_dir=self.category_dir,
                             transforms=train_transforms(resize_scale, crop_scale, is_gcn=self.is_gcn),
                             inp_name=self.inp_name)
        valid_dataset = COCO(images_dir=self.valid_path,
                             config_dir=self.category_dir,
                             transforms=valid_transforms(resize_scale, crop_scale, is_gcn=self.is_gcn),
                             inp_name=self.inp_name)

        batch_size = max(1, min(batch_size, len(train_dataset), len(valid_dataset)))

        shuffle = False
        drop_last = True
        num_workers = 32

        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset, batch_size=batch_size, num_workers=num_workers,
            drop_last=drop_last, shuffle=shuffle
        )
        valid_loader = torch.utils.data.DataLoader(
            dataset=valid_dataset, batch_size=batch_size, num_workers=num_workers,
            drop_last=drop_last, shuffle=shuffle
        )
        return train_loader, valid_loader


from torch.nn import Parameter


class MyWriter(object):
    def __init__(self, dir_name, stats_name='stats'):
        super(MyWriter, self).__init__()
        self.stats_dir = os.path.join(dir_name, stats_name)
        if not os.path.exists(self.stats_dir):
            os.makedirs(self.stats_dir)

    def get(self, file_name, buf_size=1, header=''):

        file = open(os.path.join(self.stats_dir, file_name), 'w', buffering=buf_size)
        writer = csv.writer(file)

        if len(header) != 0:
            writer.writerow(header)
        return writer


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


class LowRankBilinearAttention(nn.Module):
    def __init__(self, dim1, dim2, att_dim=2048):
        super().__init__()
        self.linear1 = nn.Linear(dim1, att_dim, bias=False)
        self.linear2 = nn.Linear(dim2, att_dim, bias=False)
        self.hidden_linear = nn.Linear(att_dim, att_dim)
        self.target_linear = nn.Linear(att_dim, 1)

        self.tanh = nn.Tanh()

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x1, x2):
        _x1 = self.linear1(x1).unsqueeze(dim=1)
        _x2 = self.linear2(x2).unsqueeze(dim=2)

        t = self.hidden_linear(self.tanh(_x1 * _x2))

        t = self.target_linear(t).squeeze(-1)

        alpha = self.softmax(t)
        label_repr = torch.bmm(alpha, x1)

        return label_repr, alpha


class EntropyLoss(nn.Module):
    def __init__(self, margin=0.0) -> None:
        super().__init__()
        self.margin = margin

    def forward(self, x, eps=1e-7):
        x = x * torch.log(x + eps)

        en = -1 * torch.sum(x, dim=-1)

        en = torch.mean(en)
        return en


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = Parameter(torch.Tensor(in_features, out_features))

        if bias:

            self.bias = Parameter(torch.Tensor(out_features))
        else:

            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):

        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def init_parameters_by_kaiming(self):
        torch.nn.init.kaiming_normal_(self.weight.data)
        if self.bias is not None:
            torch.nn.init.kaiming_normal_(self.bias.data)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)

        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'


def gen_adjs(A):
    batch_size = A.size(0)
    adjs = torch.zeros_like(A)
    for i in range(batch_size):
        D = torch.pow(A[i].sum(1).float(), -0.5)

        D = torch.diag(D)
        adj = torch.matmul(torch.matmul(A[i], D).t(), D)
        adjs[i] = adj
    return adjs


def kmeans(
        X,
        num_clusters,
        distance='euclidean',
        tol=1e-4,
        initial_state=None,
        scene_cnts=None
):
    if distance == 'euclidean':
        pairwise_distance_function = pairwise_distance
    elif distance == 'cosine':
        pairwise_distance_function = pairwise_cosine
    else:
        raise NotImplementedError

    device = X.device
    initial_state = initial_state.to(device)

    iteration = 0

    dis = pairwise_distance_function(X, initial_state)

    choice_cluster = torch.argmin(dis, dim=1)

    for index in range(num_clusters):
        selected = torch.nonzero(choice_cluster == index).squeeze().to(device)

        selected = torch.index_select(X, 0, selected)

        if len(selected) == 0:
            continue

        w1 = scene_cnts[index]
        weight = w1 + len(selected)

        initial_state[index] = (w1 * initial_state[index] + selected.sum(dim=0)) / weight

    return choice_cluster.to(device), initial_state


def pairwise_distance(data1, data2):
    A = data1.unsqueeze(dim=1)

    B = data2.unsqueeze(dim=0)

    dis = (A - B) ** 2.0

    dis = dis.sum(dim=-1).squeeze()
    return dis


def pairwise_cosine(data1, data2, device=torch.device('cpu')):
    data1, data2 = data1.to(device), data2.to(device)

    A = data1.unsqueeze(dim=1)

    B = data2.unsqueeze(dim=0)

    A_normalized = A / A.norm(dim=-1, keepdim=True)
    B_normalized = B / B.norm(dim=-1, keepdim=True)

    cosine = A_normalized * B_normalized

    cosine_dis = 1 - cosine.sum(dim=-1).squeeze()
    return cosine_dis


class ResnetKmeans(nn.Module):

    def __init__(self, model, img_size=448, embed_dim=300, feat_dim=2048, att_dim=1024, num_scenes=6,
                 gcn_middle_dim=1024, out_channels=2048, num_classes=80):

        super(ResnetKmeans, self).__init__()

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

        self.centers = torch.Tensor(num_scenes, feat_dim)

        torch.nn.init.kaiming_uniform_(self.centers, a=math.sqrt(5))

        self.total_scene_cnts = [0] * num_scenes

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
            'comat': comats,
            'scene_indices': scene_ids_x
        }

    def get_config_optim(self, lr=0.1, lrp=0.1):
        return [
            {'params': self.features.parameters(), 'lr': lr * lrp},
            {'params': self.gc1.parameters(), 'lr': lr},
            {'params': self.gc2.parameters(), 'lr': lr},
            {'params': self.attention.parameters(), 'lr': lr},
            {'params': self.fc.parameters(), 'lr': lr},

            {'params': self.classifier.parameters(), 'lr': lr}
        ]


import torchvision.models as torch_models

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--num_scenes', default=4, type=int)
parser.add_argument('--device', default='cuda:0', type=str)

args = parser.parse_args()

num_scenes = args.num_scenes
device = args.device

epochs = 40
batch_size = 8

lr, lrp = 0.0001, 0.1
num_classes = 80

stats_dir = f'stats_num_scenes_{num_scenes}'
my_writer = MyWriter(dir_name=os.getcwd(), stats_name=stats_dir)

client_header = ['epoch', 'OP', 'OR', 'OF1', 'CP', 'CR', 'CF1', 'OP_3', 'OR_3', 'OF1_3', 'CP_3', 'CR_3', 'CF1_3', 'map']

train_writer = my_writer.get("train.csv", header=client_header)
valid_writer = my_writer.get("valid.csv", header=client_header)

model = torch_models.resnet101(pretrained=True, num_classes=1000)
model = ResnetKmeans(model, num_scenes=num_scenes, num_classes=num_classes).to(device)

optimizer = torch.optim.AdamW(model.get_config_optim(lr=lr, lrp=lrp), lr=lr, weight_decay=1e-4)

criterion = AsymmetricLossOptimized().to(device)

ap_meter = AveragePrecisionMeter(difficult_examples=False)

dataset = "coco"
inp_name = f'{dataset}_glove_word2vec.pkl'

category_dir = f'/data/projects/fate/my_practice/dataset/{dataset}'
train_path = '/data/projects/dataset/clustered_dataset/client1/train'
valid_path = '/data/projects/dataset/clustered_dataset/client1/val'

dataset_loader = DatasetLoader(category_dir, train_path, valid_path, inp_name)
train_loader, valid_loader = dataset_loader.get_loaders(batch_size)

sigmoid_func = torch.nn.Sigmoid()

for epoch in range(epochs):

    scene_images = dict()
    for i in range(num_scenes):
        scene_images[i] = []

    ap_meter.reset()
    model.train()
    for train_step, ((features, inp), target) in enumerate(train_loader):
        features = features.to(device)
        target = target.to(device)
        inp = inp.to(device)
        output = model(features, inp, y=target)
        predicts = output['output']

        scene_indices = output['scene_indices']

        indices = list(train_loader.batch_sampler)[train_step]
        for i in range(len(indices)):
            index = indices[i]
            scene_images[scene_indices[i].item()].append(train_loader.dataset.img_list[index])

        ap_meter.add(predicts.data, target)

        objective_loss = criterion(sigmoid_func(predicts), target)

        overall_loss = objective_loss

        optimizer.zero_grad()

        overall_loss.backward()

        optimizer.step()
    if (epoch + 1) % 4 == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.9
    mAP, _ = ap_meter.value()
    mAP *= 100

    OP, OR, OF1, CP, CR, CF1 = ap_meter.overall()
    OP_k, OR_k, OF1_k, CP_k, CR_k, CF1_k = ap_meter.overall_topk(3)
    metrics = [OP, OR, OF1, CP, CR, CF1, OP_k, OR_k, OF1_k, CP_k, CR_k, CF1_k, mAP.item()]
    train_writer.writerow([epoch] + metrics)

    json_path = f'{stats_dir}/scene_images_{epoch}.json'
    json_content = json.dumps(scene_images, indent=4)
    with open(json_path, 'w') as json_file:
        json_file.write(json_content)

    model.eval()
    ap_meter.reset()
    with torch.no_grad():
        for validate_step, ((features, inp), target) in enumerate(valid_loader):
            features = features.to(device)
            inp = inp.to(device)
            target = target.to(device)

            output = model(features, inp, y=target)
            predicts = output['output']

            ap_meter.add(predicts.data, target)

    mAP, _ = ap_meter.value()
    mAP *= 100
    OP, OR, OF1, CP, CR, CF1 = ap_meter.overall()
    OP_k, OR_k, OF1_k, CP_k, CR_k, CF1_k = ap_meter.overall_topk(3)
    metrics = [OP, OR, OF1, CP, CR, CF1, OP_k, OR_k, OF1_k, CP_k, CR_k, CF1_k, mAP.item()]
    valid_writer.writerow([epoch] + metrics)
