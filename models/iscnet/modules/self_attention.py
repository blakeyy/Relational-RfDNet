# Self attention.

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.registers import MODULES

def hw_flatten(x):
    # Input shape x: [BATCH, HEIGHT, WIDTH, CHANNELS]
    # flat the feature volume across the width and height dimensions 
    x_shape = x.shape
    return torch.reshape(x, [x_shape[0], -1, x_shape[-1]]) # return [BATCH, W*H, CHANNELS]

class SkipParameter(nn.Module):
    def __init__(self, init_value=0.01):
        super(SkipParameter, self).__init__()
        self.skip_value = nn.Parameter(torch.ones(1)) # requires_grad is True by default for Parameter
        nn.init.constant_(self.skip_value, init_value)

    def forward(self, f_a, concat):
        return f_a + self.skip_value * concat

@MODULES.register_module    
class SelfAttention(nn.Module):
    def __init__(self, cfg, optim_spec=None):
        super(SelfAttention, self).__init__()
        '''Optimizer parameters used in training'''
        self.optim_spec = optim_spec
        self.cfg = cfg
        
        '''Parameters'''
        feat_dim = self.cfg.config['model']['self_attention']['appearance_feature_dim']
        self.layer = feat_dim//4

        '''Modules'''
        self.gamma = nn.Parameter(torch.ones(1)) # requires_grad is True by default for Parameter
        nn.init.constant_(self.gamma, 0.0)
        #self.skip = SkipParameter()

        self.F = torch.nn.Conv2d(feat_dim,self.layer, kernel_size=1, padding=0, stride=1, bias=False)
        self.G = torch.nn.Conv2d(feat_dim,self.layer, kernel_size=1, padding=0, stride=1, bias=False)
        self.H = torch.nn.Conv2d(feat_dim, feat_dim, kernel_size=1, padding=0, stride=1, bias=False)

        #self.mlp = nn.Sequential(nn.Conv1d(256,128,1), \
        #                            nn.BatchNorm1d(128), \
        #                            nn.ReLU(), \
        #                            nn.Conv1d(128,64,1))

    def forward(self, inputs):
        input, nms_feat = inputs
        if nms_feat is not None:
            feat = nms_feat
        else:
            feat = input

        
        #print(inputs.shape)
        feat = feat.unsqueeze(-1)
        f = self.F(feat)
        g = self.G(feat)
        h = self.H(feat)

        B, C, N, _ = h.shape

        f = f.transpose(dim0=1,dim1=3)
        g = g.transpose(dim0=1,dim1=3)
        h = h.transpose(dim0=1,dim1=3)

        s = torch.matmul(hw_flatten(g), hw_flatten(f).transpose(dim0=1,dim1=2))  # # [bs, N, N]

        beta = F.softmax(s, dim=-1)  # attention map

        ##### visualize
        #attention_map = []
        #for i in range(beta.shape[1]):
        #    attention_map.append("Proposal " + str(i) + ": " + str(torch.topk(beta[0][i], 10, dim=-1)))
        #end_points['attention_map'] = attention_map
        ######
        
        o = torch.matmul(beta, hw_flatten(h))   # [B, N, N]*[B, N, c]->[B, N, c]
        o = torch.reshape(o, shape=[B, C, N])  # [B, C, N]
        
        x = self.gamma * o + input
        #x = self.skip(inputs, o)
        
        #x = self.skip(inputs, concat)
        #x = inputs + self.dropout(self.group_norm(o))
        #x = input + o
        #x = inputs + self.group_norm(concat)
        #x = inputs + concat
        #x = self.mlp(o)
        return x