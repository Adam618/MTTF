import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossFusion(nn.Module):
    def __init__(self, cross_attention):
        super(CrossFusion, self).__init__()
        self.cross_attention = cross_attention
        
    def forward(self, x, x_mask=None, cross_mask=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)

        if self.norm is not None:
            x = self.norm(x)

        return x