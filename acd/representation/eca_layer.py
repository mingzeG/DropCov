import torch
from torch import nn
from torch.nn.parameter import Parameter

class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        # y = self.sigmoid(y)
        y = self.softmax(y)

        return x * y.expand_as(x) + x, y

class eca_layer_drop(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3, mask_style = 'larger', p = 0.5):
        super(eca_layer_drop, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()
        self.mask_style = mask_style
        self.drop_rate = p
        
    def dy_drop(self, y):
        b, c,_,_ = y.size()       
        # mask = torch.zeros(size = y.shape) # cpu
        mask = torch.zeros(size = y.shape).cuda() # gpu
        sort_sum, index = torch.sort(y, dim=1, descending= True)
        if self.mask_style == 'uniform': 
            mask_index = index[:,::2] 
            # mask_index = index[:,::3] 
        elif self.mask_style == 'larger':
            mask_index = index[:, :int(c * self.drop_rate)] 
        else:
            assert 0  , 'please choose from (uniform, larger)' 
        mask = mask.scatter_(1, mask_index, 1)
        y = y.mul(mask)
        y = y.view(b, c, 1, 1)
        return y     


    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)
        
        y = self.dy_drop(y)

        return x * y.expand_as(x)
