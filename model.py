import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


# ref: https://blog.csdn.net/immc1979/article/details/128324029
class ResNet18_32x32(nn.Module):
  
  def __init__(self, num_classes=10, pretrained=True):
    super().__init__()

    self.m = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
    self.m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=3//2, bias=False)
    self.m.maxpool = nn.MaxPool2d(1, 1, 0)
    self.m.fc = nn.Linear(self.m.fc.in_features, num_classes)
  
  def forward(self, x):
    return self.m(x)
