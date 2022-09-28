import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import numpy as np

class Covariance(nn.Module):
    def __init__(self, 
                cov_type='norm',
                remove_mean=True,
                dimension_reduction=None,
                input_dim=2048,
        ):
        super(Covariance, self).__init__()
        self.cov_type = cov_type
        self.remove_mean = remove_mean
        self.dr = dimension_reduction
        if self.dr is not None:
            if self.cov_type == 'norm':
                self.conv_dr_block = nn.Sequential(
                    nn.Conv2d(input_dim, self.dr[0], kernel_size=1, stride=1, bias=False),
                    nn.BatchNorm2d(self.dr[0]),
                    nn.ReLU(inplace=True)
                )
            elif self.cov_type == 'cross':
                self.conv_dr_block = nn.Sequential(
                    nn.Sequential(
                        nn.Conv2d(input_dim, self.dr[0], kernel_size=1, stride=1, bias=False),
                        nn.BatchNorm2d(self.dr[0]),
                        nn.ReLU(inplace=True)
                    ),
                    nn.Sequential(
                        nn.Conv2d(input_dim, self.dr[1], kernel_size=1, stride=1, bias=False),
                        nn.BatchNorm2d(self.dr[1]),
                        nn.ReLU(inplace=True)
                    )
                )
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    

    def _remove_mean(self, x):
        _mean = F.adaptive_avg_pool2d(x, (1,1))
        x = x - _mean
        return x

    def _cov(self, x):
        # channel
        batchsize, d, h, w = x.size()
        N = h*w
        x = x.view(batchsize, d, N)
        y = (1. / N ) * (x.bmm(x.transpose(1, 2)))
        return y
    
    def _cross_cov(self, x1, x2):
        # channel
        batchsize1, d1, h1, w1 = x1.size()
        batchsize2, d2, h2, w2 = x2.size()
        N1 = h1*w1
        N2 = h2*w2
        assert batchsize1 == batchsize2
        assert N1 == N2
        x1 = x1.view(batchsize1, d1, N1)
        x2 = x2.view(batchsize2, d2, N2)

        y = (1. / N1) * (x1.bmm(x2.transpose(1, 2)))
        return y
    
    def forward(self, x, y=None):
        #import pdb;pdb.set_trace()
        if self.dr is not None:
            if self.cov_type == 'norm':
                x = self.conv_dr_block(x)
            elif self.cov_type == 'cross':
                if y is not None:
                    x = self.conv_dr_block[0](x)
                    y = self.conv_dr_block[1](y)
                else:
                    ori = x
                    x = self.conv_dr_block[0](ori)
                    y = self.conv_dr_block[1](ori)
        if self.remove_mean:
            x = self._remove_mean(x)
            if y is not None:
                y = self._remove_mean(y)               
        if y is not None:
            x = self._cross_cov(x, y)
        else:
            x = self._cov(x)
        return x

class Triuvec(Function):
     @staticmethod
     def forward(ctx, input):
         x = input
         batchSize = x.data.shape[0]
         dim = x.data.shape[1]
         dtype = x.dtype
         x = x.reshape(batchSize, dim*dim)
         I = torch.ones(dim,dim).triu().reshape(dim*dim)
         index = I.nonzero(as_tuple=False)
         y = torch.zeros(batchSize,int(dim*(dim+1)/2),device = x.device).type(dtype)
         y = x[:,index]
         ctx.save_for_backward(input,index)
         return y
     @staticmethod
     def backward(ctx, grad_output):
         input,index = ctx.saved_tensors
         x = input
         batchSize = x.data.shape[0]
         dim = x.data.shape[1]
         dtype = grad_output.dtype
         grad_input = torch.zeros(batchSize,dim*dim,device = x.device,requires_grad=False).type(dtype)
         grad_input[:,index] = grad_output
         grad_input = grad_input.reshape(batchSize,dim,dim)
         return grad_input
     