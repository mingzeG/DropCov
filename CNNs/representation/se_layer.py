from torch import nn 
import torch 

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# based on SElayer weight dy_dropout
class SELayer_drop(nn.Module):
    def __init__(self, channel, reduction=16, mask_style = 'larger', p = 0.5):
        super(SELayer_drop, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        self.mask_style = mask_style
        self.drop_rate = p
    # structural dynamic dropout    
    def dy_drop(self, y):
        b, c = y.size()       
        # mask = torch.zeros(size = y.shape) # cpu
        mask = torch.zeros(size = y.shape).cuda() # gpu
        sort_sum, index = torch.sort(y, dim=1, descending= True)
        if self.mask_style == 'uniform': 
            # mask_index = index[:,::2] 
            mask_index = index[:,::3] 
        elif self.mask_style == 'larger':
            mask_index = index[:, :int(c * self.drop_rate)] 
        else:
            assert 0  , 'please choose from (uniform, larger)' 
        mask = mask.scatter_(1, mask_index, 1)
        y = y.mul(mask)
        y = y.view(b, c, 1, 1)
        return y     

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y)
        y = self.dy_drop(y)
        return x * y.expand_as(x)
