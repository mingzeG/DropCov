import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import numpy as np
from .covariance import Covariance
from ..normalization import svPN, MPN, ePN, scaleNorm

class MultiHeadPooling(nn.Module):
    def __init__(self, dim, num_heads=8, wr_dim=16, qkv_bias=False, normalization={}):
        super().__init__()
        self.num_heads = num_heads 
        head_dim = dim // num_heads

        self.scale = head_dim ** -0.5

        self.wr = nn.Linear(dim, wr_dim*num_heads*3, bias=qkv_bias) # modified
        # self.wr = nn.Linear(dim, wr_dim*num_heads*2, bias=qkv_bias)
        self.wr_dim = wr_dim
        self.dropout=nn.Dropout(0.5)
        self.drop=nn.Dropout2d(0.2)
        self.cov = Covariance()
        # norm_type  = normalization['type']S
        norm_type  = None
        normalization.pop('type')
        if norm_type == 'svPN':
            self.norm = svPN(**normalization)
        elif norm_type == 'sqrt_d':
            self.norm = scaleNorm(factor=self.wr_dim**-0.5)
        elif norm_type == 'sqrt_t':
            self.norm = scaleNorm(factor=self.wr_dim**-0.5, is_learnable=True)
        elif norm_type == 'ePN':
            self.norm = ePN()
        elif norm_type == 'LN':
            self.norm = nn.LayerNorm(self.wr_dim)
        elif norm_type == None:
            self.norm = nn.Identity()
        self.output_dim = num_heads * (wr_dim ** 2)

    def forward(self, x):
        B, N, C = x.shape
        # qk = self.wr(x).reshape(B, N, 2, self.num_heads, self.wr_dim).permute(2, 0, 3, 1, 4)
        qk = self.wr(x).reshape(B, N, 3, self.num_heads, self.wr_dim).permute(2, 0, 3, 1, 4)
        q, k = qk[0], qk[1]

        q = q.reshape(B*self.num_heads, N, self.wr_dim)
        k = k.reshape(B*self.num_heads, N, self.wr_dim)
        mask = torch.ones(size=(B*self.num_heads, N, self.wr_dim)).to(x)
        mask = self.drop(mask).to(x)
        q = q.mul(mask).to(x)
        k = k.mul(mask).to(x)
        # q=self.drop(q)
        # k=self.drop(k)
        # cov
        x = self.cov(q, k)
        x = self.dropout(x)
        #x = self.norm(x)
        x = x.view(B, -1)
        return x
