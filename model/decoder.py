import torch
from torch import nn
import torch.nn.functional as F

class predBlock1(nn.Module):
  expansion = 4

  def __init__(self, inplanes, planes, stride=1, downsample=None):
    super(predBlock1, self).__init__()
    
    self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size=1, stride=stride, bias=False) # change
    self.bn1 = nn.BatchNorm2d(inplanes)
    
    self.conv2 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, # change
                 padding=1, bias=False)
    self.bn2 = nn.BatchNorm2d(planes)
    
    self.conv3 = nn.Conv2d(planes, planes, kernel_size=1, bias=False)
    self.bn3 = nn.BatchNorm2d(planes)
    self.relu = nn.ReLU(inplace=True)
    
    self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes,
                          kernel_size=3, stride=1,padding=1, bias=False),
                nn.BatchNorm2d(planes),
            )
    self.stride = stride
  def forward(self, x):
    residual = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)
    out = self.relu(out)

    out = self.conv3(out)
    out = self.bn3(out)

    if self.downsample is not None:
      residual = self.downsample(x)

    out += residual
    out = self.relu(out)

    return out

class predBlock2(nn.Module):
  expansion = 4

  def __init__(self, inplanes, planes, stride=1, downsample=None):
    super(predBlock2, self).__init__()
    
    self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False) # change
    self.bn1 = nn.BatchNorm2d(planes)
    
    self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, # change
                 padding=1, bias=False)
    self.bn2 = nn.BatchNorm2d(planes)
    
    self.conv3 = nn.Conv2d(planes, planes, kernel_size=1, bias=False)
    self.bn3 = nn.BatchNorm2d(planes)
    self.relu = nn.ReLU(inplace=True)
    
    self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes,
                          kernel_size=3, stride=1, padding=1,bias=False),
                nn.BatchNorm2d(planes),
            )
    self.stride = stride
  def forward(self, x):
    residual = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)
    out = self.relu(out)

    out = self.conv3(out)
    out = self.bn3(out)

    if self.downsample is not None:
      residual = self.downsample(x)

    out += residual
    out = self.relu(out)

    return out
class predBlock3(nn.Module):
  expansion = 4

  def __init__(self, inplanes, planes, stride=1, downsample=None):
    super(predBlock3, self).__init__()
    
    self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False) # change
    self.bn1 = nn.BatchNorm2d(planes)
    
  def forward(self, x):


    out = self.conv1(x)
    out = self.bn1(out)

    return out

class decoder_conv(nn.Module):
  def __init__(self, inplanes, outplane, stride=1, downsample=None):
    super(decoder_conv, self).__init__()
    self.layer1=predBlock1(inplanes,outplane)
    self.layer2=predBlock2(inplanes*2,inplanes)
    self.layer3=predBlock3(inplanes,outplane)

  def forward(self, x):


    out = self.layer1(x)
    # out = self.layer2(out)
    # out = self.layer3(out)

    return out
    
