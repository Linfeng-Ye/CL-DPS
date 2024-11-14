# Model.py
import torchvision.models as models
import torch
import torch.nn as nn

class BFBatchNorm2d(nn.BatchNorm2d):
	def __init__(self, num_features, eps=1e-5, momentum=0.1, use_bias = False, affine=True):
		super(BFBatchNorm2d, self).__init__(num_features, eps, momentum)

		self.use_bias = use_bias

	def forward(self, x):
		self._check_input_dim(x)
		y = x.transpose(0,1)
		return_shape = y.shape
		y = y.contiguous().view(x.size(1), -1)
		if self.use_bias:
			mu = y.mean(dim=1)
		sigma2 = y.var(dim=1)

		if self.training is not True:
			if self.use_bias:        
				y = y - self.running_mean.view(-1, 1)
			y = y / ( self.running_var.view(-1, 1)**0.5 + self.eps)
		else:
			if self.track_running_stats is True:
				with torch.no_grad():
					if self.use_bias:
						self.running_mean = (1-self.momentum)*self.running_mean + self.momentum * mu
					self.running_var = (1-self.momentum)*self.running_var + self.momentum * sigma2
			if self.use_bias:
				y = y - mu.view(-1,1)
			y = y / (sigma2.view(-1,1)**.5 + self.eps)

		if self.affine:
			y = self.weight.view(-1, 1) * y
			if self.use_bias:
				y += self.bias.view(-1, 1)
		return y.view(return_shape).transpose(0,1)


def turn_off_bias(model):
    for layer in model.modules():
        # breakpoint()
        if hasattr(layer, 'bias') and layer.bias is not None:
            # Set bias to None or turn it off
            layer_name = type(layer).__name__
            if  layer_name == 'BatchNorm2d':
                eps = layer.eps
                num_features = layer.num_features
                momentum = layer.momentum
                layer = BFBatchNorm2d(num_features=num_features, eps=eps, momentum=momentum, use_bias = False, affine=True)

class Resnet50(nn.Module):
    '''`
    Resnet 50.

    Args:
        dim (int): Dimension of the last layer.
    '''
    def __init__(self, dim=128, bias=False):
        super(Resnet50, self).__init__()
        self.resnet = models.wide_resnet50_2(pretrained=False)
        self.resnet.fc = nn.Linear(2048, dim, bias=True)
        # if not bias:
        #     print("Turn Off Bias")
        #     turn_off_bias(self.resnet)
    def forward(self, x):
        # See note [TorchScript super()]
        
################################################################
        #x = self.resnet.conv1(x)
       # x = self.resnet.bn1(x)
       # x = self.resnet.relu(x)
       # x = self.resnet.maxpool(x)

        #x = self.resnet.layer1(x)
        #x = self.resnet.layer2(x)
        #x = self.resnet.layer3(x)
        #x = self.resnet.layer4(x)
        out = self.resnet(x)
        # out = self.resnet.avgpool(x)
        # x = torch.flatten(x, 1)
        ##out = self.resnet.fc(x)
        norm = torch.norm(out, p='fro', dim=1, keepdim=True)
        #color = torch.sigmoid(color)
        return out / norm
    
    
if __name__ == "__main__":
    print("Test Code!")
    model = Resnet50()
