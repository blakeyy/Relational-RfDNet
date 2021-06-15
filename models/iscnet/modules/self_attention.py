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

@MODULES.register_module    
class SelfAttention(nn.Module):
    def __init__(self, cfg, optim_spec=None):
        super(SelfAttention, self).__init__()
        '''Optimizer parameters used in training'''
        self.optim_spec = optim_spec
        self.cfg = cfg
        
        '''Parameters'''
        dim = 128
        layer = dim//4

        '''Modules'''
        self.gamma = nn.Parameter(torch.ones(1)) # requires_grad is True by default for Parameter
        nn.init.constant_(self.gamma, 0.3)

        self.F = torch.nn.Conv2d(dim,layer, kernel_size=1, padding=0, stride=1, bias=False)
        self.G = torch.nn.Conv2d(dim,layer, kernel_size=1, padding=0, stride=1, bias=False)
        self.H = torch.nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, bias=False)
    
    def forward(self, inputs):
        inputs = inputs.unsqueeze(-1)
        f = self.F(inputs)
        g = self.G(inputs)
        h = self.H(inputs)

        f = f.transpose(dim0=1,dim1=3)
        g = g.transpose(dim0=1,dim1=3)
        h = h.transpose(dim0=1,dim1=3)

        s = torch.matmul(hw_flatten(g), hw_flatten(f).transpose(dim0=1,dim1=2))  # # [bs, N, N]

        beta = F.softmax(s, dim=-1)  # attention map

        o = torch.matmul(beta, hw_flatten(h))   # [bs, N, N]*[bs, N, c]->[bs, N, c]
        
        inputs = torch.squeeze(inputs, dim=-1)
        o = torch.reshape(o, shape=inputs.shape)  # [bs, h, w, C]
        x = self.gamma * o + inputs
        
        return x