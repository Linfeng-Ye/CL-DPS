from abc import ABC, abstractmethod
import torch
import torch.nn as nn
__CONDITIONING_METHOD__ = {}
from model import Resnet50
from torchvision import transforms, utils

import copy
class ProjectionHead(nn.Module):
    def __init__(self):
        super(ProjectionHead, self).__init__()
        self.linear1 = nn.Linear(2048, 2048)
        self.activation = nn.ReLU()
        self.linear3 = nn.Linear(2048, 3)
        
        self.linear2 = nn.Linear(2048, 128)
        
        self.Apooling = None
    def forward(self, x):
        fx = self.Apooling(x)
        fx = torch.flatten(fx, 1)
        fx = self.linear2(fx)
        xT = torch.transpose(x, 1, 3)
        xT = self.linear1(xT)
        xT = self.activation(xT)
        xT = self.linear3(xT)
        xT = torch.transpose(xT, 1, 3)
        
        return [fx, xT]

projectionHead = ProjectionHead()
encoder = Resnet50(dim=128, bias=True)
projectionHead.Apooling = encoder.resnet.avgpool
# avgpool = copy.deepcopy(encoder.resnet.avgpool)
encoder.resnet.avgpool=torch.nn.Identity()
encoder.resnet.fc = projectionHead
projectionHead.Apooling = encoder.resnet.avgpool

# encoder.resnet.fc = projectionHead
# breakpoint()
Model = nn.DataParallel(encoder).cuda()

ckpt = torch.load("/home/deponce/scratch/diffusion-posterior-sampling-main-patch-color-align-WRN-correct-input/DNNckpt/ckpt_400.pkl")
Model.load_state_dict(ckpt['encoder'])

Model.module.resnet.fc = nn.Identity()
Model.module.resnet.avgpool = nn.Identity()

Model.eval()
Model.cuda()
print("Load model", flush=True)
# breakpoint()
KernelSize = 128
Stride=8
Unfold = torch.nn.Unfold(KernelSize, stride=Stride)

MEAN = (0.5183399319648743, 0.42324939370155334, 0.3783569931983948)                                                                                                                                                                                                                                                                                                                                    
STD = (0.2805287837982178, 0.25457799434661865, 0.2554226815700531)
Normalize = transforms.Normalize(MEAN, STD) 

invNormalize = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.5, 1/0.5, 1/0.5 ]),
                                transforms.Normalize(mean = [ -0.5, -0.5, -0.5],
                                                     std = [ 1., 1., 1. ]),
                               ])

def register_conditioning_method(name: str):
    def wrapper(cls):
        if __CONDITIONING_METHOD__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __CONDITIONING_METHOD__[name] = cls
        return cls
    return wrapper

def get_conditioning_method(name: str, operator, noiser, **kwargs):
    if __CONDITIONING_METHOD__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined!")
    return __CONDITIONING_METHOD__[name](operator=operator, noiser=noiser, **kwargs)

    
class ConditioningMethod(ABC):
    def __init__(self, operator, noiser, **kwargs):
        self.operator = operator
        self.noiser = noiser
    def project(self, data, noisy_measurement, **kwargs):
        return self.operator.project(data=data, measurement=noisy_measurement, **kwargs)
    
    def grad_and_value(self, x_prev, x_0_hat, measurement, **kwargs):
        if self.noiser.__name__ == 'gaussian':
            difference = measurement - self.operator.forward(x_0_hat, **kwargs)
            norm = torch.linalg.norm(difference)
            norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]
            
            # Unfoldmeasurement = Unfold(measurement)
            # Unfoldmeasurement = Unfoldmeasurement.transpose(1,2)
            # Unfoldmeasurement = Unfoldmeasurement.reshape(-1,3,KernelSize, KernelSize)
            # # Unfoldmeasurement = measurement
            # F_y = Model(Normalize(invNormalize(Unfoldmeasurement))).detach()
            # # breakpoint()
            # Unfoldx_0_hat = Unfold(x_0_hat)
            # Unfoldx_0_hat = Unfoldx_0_hat.transpose(1,2)
            # Unfoldx_0_hat = Unfoldx_0_hat.reshape(-1,3,KernelSize, KernelSize)
            # # Unfoldx_0_hat = x_0_hat
            # F_x = Model(Normalize(invNormalize(Unfoldx_0_hat)))

            # # breakpoint()
            # # difference = F_y-F_x
            # # norm = torch.linalg.norm(difference) #/ measurement.abs()
            # # norm = -torch.inner(F_y, F_x)
            # # breakpoint()
            # norm = -torch.mean(torch.sum((F_y*F_x),1))
            # # norm = -torch.sum((F_y*F_x))
            # # norm = -torch.mean(torch.sum((torch.log(F_y*F_x)),1))
            # # difference = measurement - self.operator.forward(x_0_hat, **kwargs)
            # # norm = torch.linalg.norm(difference)
            # norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]
        


        elif self.noiser.__name__ == 'poisson':
            Ax = self.operator.forward(x_0_hat, **kwargs)
            difference = measurement-Ax
            norm = torch.linalg.norm(difference) / measurement.abs()
            norm = norm.mean()
            norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]

        else:
            raise NotImplementedError
             
        return norm_grad, norm
   
    @abstractmethod
    def conditioning(self, x_t, measurement, noisy_measurement=None, **kwargs):
        pass
    
@register_conditioning_method(name='vanilla')
class Identity(ConditioningMethod):
    # just pass the input without conditioning
    def conditioning(self, x_t):
        return x_t
    
@register_conditioning_method(name='projection')
class Projection(ConditioningMethod):
    def conditioning(self, x_t, noisy_measurement, **kwargs):
        x_t = self.project(data=x_t, noisy_measurement=noisy_measurement)
        return x_t


@register_conditioning_method(name='mcg')
class ManifoldConstraintGradient(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.scale = kwargs.get('scale', 1.0)
        
    def conditioning(self, x_prev, x_t, x_0_hat, measurement, noisy_measurement, **kwargs):
        # posterior sampling
        norm_grad, norm = self.grad_and_value(x_prev=x_prev, x_0_hat=x_0_hat, measurement=measurement, **kwargs)
        x_t -= norm_grad * self.scale
        
        # projection
        x_t = self.project(data=x_t, noisy_measurement=noisy_measurement, **kwargs)
        return x_t, norm
        
@register_conditioning_method(name='ps')
class PosteriorSampling(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.scale = kwargs.get('scale', 1.0)

    def conditioning(self, x_prev, x_t, x_0_hat, measurement, **kwargs):
        norm_grad, norm = self.grad_and_value(x_prev=x_prev, x_0_hat=x_0_hat, measurement=measurement, **kwargs)
        x_t -= norm_grad * self.scale
        return x_t, norm
        
@register_conditioning_method(name='ps+')
class PosteriorSamplingPlus(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.num_sampling = kwargs.get('num_sampling', 5)
        self.scale = kwargs.get('scale', 1.0)

    def conditioning(self, x_prev, x_t, x_0_hat, measurement, **kwargs):
        norm = 0
        for _ in range(self.num_sampling):
            # TODO: use noiser?
            x_0_hat_noise = x_0_hat + 0.05 * torch.rand_like(x_0_hat)
            difference = measurement - self.operator.forward(x_0_hat_noise)
            norm += torch.linalg.norm(difference) / self.num_sampling
        
        norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]
        x_t -= norm_grad * self.scale
        return x_t, norm
