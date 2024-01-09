import torch
import torch.nn as nn
import torch.nn.functional as F

from backend.gcn.models.graph_convolution import GraphConvolution
from backend.gcn.utils import gen_adjs



class EntropyLoss(nn.Module):
    def __init__(self, margin=0.0) -> None:
        super().__init__()
        self.margin = margin

    
    
    def forward(self, x, eps=1e-7):
        
        x = x * torch.log(x + eps)
        
        en = -1 * torch.sum(x, dim=-1)
        
        
        en = torch.mean(en)
        return en


class LowRankBilinearAttention(nn.Module):
    def __init__(self, dim1, dim2, att_dim=2048):
        """

        Parameters
        ----------
        dim1: 图像的特征大小
        dim2: 标签的特征大小
        att_dim: 注意力网络的大小
        """
        super().__init__()
        self.linear1 = nn.Linear(dim1, att_dim, bias=False)
        self.linear2 = nn.Linear(dim2, att_dim, bias=False)
        self.hidden_linear = nn.Linear(att_dim, att_dim)
        self.target_linear = nn.Linear(att_dim, 1)
        
        self.tanh = nn.Tanh()
        
        
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x1, x2):
        """
        执行前向传播
        Parameters
        ----------
        x1: 维度(B, num_pixels, dim1)，每个像素点都有一个特征向量，dim1应该就是通道吧？
        x2: 维度(B, num_labels, dim2)，每个标签都有一个特征向量，每个批次的标签向量都是一样的？
        """
        _x1 = self.linear1(x1).unsqueeze(dim=1)  
        _x2 = self.linear2(x2).unsqueeze(dim=2)  
        
        t = self.hidden_linear(self.tanh(_x1 * _x2))
        
        t = self.target_linear(t).squeeze(-1)
        
        alpha = self.softmax(t)
        label_repr = torch.bmm(alpha, x1)
        
        return label_repr, alpha


class SALGL(nn.Module):
    
    def __init__(self, model, feat_dim=2048, att_dim=1024, num_scenes=6, in_channels=300,
                 out_channels=2048, num_classes=80, comat_smooth=False):

        super(SALGL, self).__init__()
        
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

        
        self.pooling = nn.MaxPool2d(14, 14)
        
        self.scene_linear = nn.Linear(feat_dim, num_scenes, bias=False)
        torch.nn.init.kaiming_normal_(self.scene_linear.weight)
        
        self.gc1 = GraphConvolution(in_channels, 1024)
        self.gc2 = GraphConvolution(1024, out_channels)
        self.relu = nn.LeakyReLU(0.2)
        
        self.register_buffer('comatrix', torch.zeros((num_scenes, num_classes, num_classes)))

        self.entropy = EntropyLoss()
        
        self.max_en = self.entropy(torch.tensor([1 / num_scenes] * num_scenes))

        
        
        

    
    def comatrix2prob(self):
        
        
        
        
        comat = self.comatrix
        
        temp = torch.diagonal(comat, dim1=1, dim2=2).unsqueeze(-1)
        
        comat = comat / (temp + 1e-8)
        
        
        for i in range(self.num_classes):
            comat[:, i, i] = 1
        return comat

    """
    x: 图像特征
    y: 图像标签集合
    inp: 标签嵌入
    """

    def forward(self, x, inp, y=None):
        img_feats = self.features(x)
        img_feats = self.pooling(img_feats)
        
        img_feats = img_feats.view(img_feats.size(0), -1)  

        scene_scores = self.scene_linear(img_feats)  
        
        if self.training:
            _scene_scores = scene_scores
            _scene_probs = F.softmax(_scene_scores, dim=-1)
            
            
            batch_comats = torch.bmm(y.unsqueeze(2), y.unsqueeze(1))
            
            for i in range(_scene_probs.shape[0]):
                
                if self.comat_smooth:
                    prob = _scene_probs[i].unsqueeze(-1).unsqueeze(-1)  
                    comat = batch_comats[i].unsqueeze(0)  
                    
                    self.comatrix += prob * comat  
                else:
                    max_scene_id = torch.argmax(_scene_probs[i])
                    
                    self.comatrix[max_scene_id] += batch_comats[i]
        
        scene_probs = F.softmax(scene_scores, dim=-1)
        
        sample_en = self.entropy(scene_probs)
        
        _scene_probs = torch.mean(scene_probs, dim=0)
        
        
        batch_en = (self.max_en - self.entropy(_scene_probs)) * 1

        
        
        

        
        if not self.comat_smooth:
            comats = self.comatrix2prob()
            
            indices = torch.argmax(scene_probs, dim=-1)
            
            comats = torch.index_select(comats, dim=0, index=indices)
        else:  
            _scene_probs = scene_probs.unsqueeze(-1).unsqueeze(-1)  
            comats = self.comatrix2prob().unsqueeze(0)  
            comats = _scene_probs * comats  
            
            comats = torch.sum(comats, dim=1)  

        
        
        
        
        
        adjs = gen_adjs(comats)
        
        x = self.gc1(inp, adjs)
        x = self.relu(x)  
        x = self.gc2(x, adjs)  

        
        output = torch.matmul(x, img_feats.unsqueeze(2)).squeeze(-1)
        
        
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
            {'params': self.scene_linear.parameters(), 'lr': lr}  
        ]
