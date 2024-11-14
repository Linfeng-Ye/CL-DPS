# Model.py
import torchvision.models as models
import torch
import torch.nn as nn

class Resnet50(nn.Module):
    '''
    Resnet 50.

    Args:
        dim (int): Dimension of the last layer.
    '''
    def __init__(self, dim=128):
        super(Resnet50, self).__init__()
        self.resnet = models.wide_resnet50_2(pretrained=False)
        self.resnet.fc = nn.Linear(2048, dim)
        
    def forward(self, x):
        # See note [TorchScript super()]
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        # out = self.resnet.avgpool(x)
        # x = torch.flatten(x, 1)
        out, color = self.resnet.fc(x)
        norm = torch.norm(out, p='fro', dim=1, keepdim=True)
        color = torch.sigmoid(color)
        return out / norm, color