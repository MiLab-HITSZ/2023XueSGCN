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

