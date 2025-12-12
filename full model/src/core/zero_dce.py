import torch
import torch.nn as nn
import torch.nn.functional as F

class C_DCE_Ratio(nn.Module):
    def __init__(self, in_channels=3, out_channels=32):
        super(C_DCE_Ratio, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        
        self.e_conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True) 
        self.e_conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=True) 
        self.e_conv3 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=True) 
        self.e_conv4 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=True) 
        self.e_conv5 = nn.Conv2d(out_channels*2, out_channels, 3, 1, 1, bias=True) 
        self.e_conv6 = nn.Conv2d(out_channels*2, out_channels, 3, 1, 1, bias=True) 
        self.e_conv7 = nn.Conv2d(out_channels*2, 3, 3, 1, 1, bias=True) 

    def forward(self, x):
        x1 = self.relu(self.e_conv1(x))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))

        x5 = self.relu(self.e_conv5(torch.cat([x3,x4],1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2,x5],1)))
        x_r = F.tanh(self.e_conv7(torch.cat([x1,x6],1)))
        
        return x_r

class ZeroDCE(nn.Module):
    def __init__(self):
        super(ZeroDCE, self).__init__()
        self.scale_factor = 1.0
        self.unet = C_DCE_Ratio()

    def enhance(self, x, r):
        x = x + r * (torch.pow(x, 2) - x)
        x = x + r * (torch.pow(x, 2) - x)
        x = x + r * (torch.pow(x, 2) - x)
        x = x + r * (torch.pow(x, 2) - x)
        x = x + r * (torch.pow(x, 2) - x)
        x = x + r * (torch.pow(x, 2) - x)
        x = x + r * (torch.pow(x, 2) - x)
        x = x + r * (torch.pow(x, 2) - x)
        return x

    def forward(self, x):
        r = self.unet(x)
        x = self.enhance(x, r)
        return x
