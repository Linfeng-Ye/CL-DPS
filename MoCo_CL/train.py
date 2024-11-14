import os
import pprint
import argparse
import yaml
import pickle

import torch.nn as nn
import torch
import torch.optim as optim

from dataloader import data_loader
import model
import util
from util import Transpose0312
from FFHQ_Dataset import FFHQ
from torch.utils.data import Dataset, DataLoader
from MotionBlur import RandomMotionBlur
from torchvision import transforms, utils
from tqdm import tqdm
from torchsummary import summary
import torchvision.transforms.functional as F

dev = 'cuda' if torch.cuda.is_available() else 'cpu'

''' ######################## < Step 1 > Parsing training arguments ######################## '''

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

parser = argparse.ArgumentParser()

# Config - Path
parser.add_argument('--dataset_root', type=str, default='/home/nas_datasets/ILSVRC/Data/CLS-LOC/train',
                    help='Root directory of dataset.')
parser.add_argument('--output_root', type=str, default='output',
                    help='Root directory of training results.')
parser.add_argument('--dataset_name', type=str, default='IMAGENET-64',
                    help='Name of dataset.')
parser.add_argument('--exp_version', type=str, default='v1',
                    help='Version of experiment.')

# Config - Hyperparameter
parser.add_argument('--batch_size', type=int, default=256,
                    help='Batch size to train an encoder.')
parser.add_argument('--lr', type=float, default=0.03,
                    help='Learning rate to train an encoder.')
parser.add_argument('--SGD_momentum', type=float, default=0.9,
                    help='Momentum of SGD optimizer to train an encoder.')
parser.add_argument('--weight_decay', type=float, default=0.0001,
                    help='Weight of L2 regularization of SGD optimizer.')
parser.add_argument('--temperature', type=float, default=0.07,
                    help='Temperature for constrastive loss.')
parser.add_argument('--momentum', type=float, default=0.999,
                    help='Momentum of momentum encoder. m in Eq.(2). It is not the momentum of SGD optimizer.')
parser.add_argument('--shuffle_bn', action='store_true',
                    help='Turn on shuffled batch norm. See Section 3.3.')

# Config - Architecture
parser.add_argument('--feature_dim', type=int, default=128,
                    help='Output dimension of last fully connected layer in encoder.')
parser.add_argument('--num_keys', type=int, default=4096,
                    help='Size of dictionary of encoded keys.')

# Config - Setting
parser.add_argument('--resize', type=int, default=84,
                    help='Image is resized to this value.')
parser.add_argument('--crop', type=int, default=64,
                    help='Image is cropped to this value. This is the final size of image transformation.')
parser.add_argument('--max_epoch', type=int, default=800,
                    help='Maximum epoch to train an encoder.')
parser.add_argument('--eval_epoch', type=int, default=10,
                    help='Frequency of evaluate an encoder.')
parser.add_argument('--plot_iter', type=int, default=1000,
                    help='Frequency of plot loss graph.')
parser.add_argument('--save_weight_epoch', type=int, default=10,
                    help='Frequency of saving weight.')
parser.add_argument('--num_workers', type=int, default=16,
                    help='Number of workers for data loader.')
parser.add_argument('--save_config', action='store_true',
                    help='Save training configuration. It requires PyYAML.')


parser.add_argument('--resume', action='store_true',
                    help='Resume training.')
parser.add_argument('--start_epoch', type=int, default=0,
                    help='Training is resumed at this epoch.')

config = parser.parse_args()

# Show config
print('\n======================= Training configuration =======================\n')
pprint.pprint(vars(config))
print('\n======================================================================\n')

# Make output directories
output_path = os.path.join(config.output_root, config.dataset_name, config.exp_version)
loss_path = os.path.join(output_path, 'loss')
weight_path = os.path.join(output_path, 'weight')
    
if not os.path.exists(loss_path):
    os.makedirs(loss_path)
if not os.path.exists(weight_path):
    os.makedirs(weight_path)
            
# Save config in yaml file
if config.save_config:
    import yaml
    args_yaml = yaml.dump(vars(config))
    config_yaml_path = os.path.join(output_path, 'config.yml')
    with open(config_yaml_path, 'w') as fp:
        yaml.dump(args_yaml, fp, default_flow_style=True)
    
    
''' ######################## < Step 2 > Create instances ######################## '''

# Build dataloader
print('\n[1 / 3]. Build data loader. Depending on your environment, this may take several minutes..')



Kernel = RandomMotionBlur(size = (61, 61))


Resize = transforms.Resize(256)
Distortion = Kernel

def forward(self, img, i=None, j=None, h=None, w=None):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped and resized.

        Returns:
            PIL Image or Tensor: Randomly cropped and resized image.
        """
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        return {F.resized_crop(img, i, j, h, w, self.size, self.interpolation, antialias=self.antialias), [i, j, h, w]}


class JointRandomResizedCrop(transforms.RandomResizedCrop):
    def __init__(self, size, scale, ratio):
        super(JointRandomResizedCrop, self).__init__(size=size, scale=scale, ratio=ratio)
    def forward(self, img, dimg):
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        img = F.resized_crop(img, i, j, h, w, self.size, self.interpolation)
        dimg = F.resized_crop(dimg, i, j, h, w, self.size, self.interpolation)
        return img, dimg


# JRC = JointRandomResizedCrop((64,64), scale=(0.045, 0.08), ratio=(0.5, 2))
JRC = JointRandomResizedCrop((64,64), scale=(0.045, 0.5), ratio=(0.5, 2))


Transform = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5)])

MEAN = (0.5183399319648743, 0.42324939370155334, 0.3783569931983948)

STD = (0.2805287837982178, 0.25457799434661865, 0.2554226815700531)

Normalize = transforms.Normalize(MEAN, STD)
# Build models
print('\n[2 / 3]. Build models.. ')


encoder = model.Resnet50(dim=config.feature_dim)


projectionHead = ProjectionHead()
projectionHead.Apooling = encoder.resnet.avgpool
encoder.resnet.avgpool=torch.nn.Identity()
encoder.resnet.fc = projectionHead

encoder = nn.DataParallel(encoder).to(dev)

momentum_encoder = model.Resnet50(dim=config.feature_dim)
MprojectionHead = ProjectionHead()
MprojectionHead.Apooling = momentum_encoder.resnet.avgpool
momentum_encoder.resnet.avgpool=torch.nn.Identity()
momentum_encoder.resnet.fc = MprojectionHead
momentum_encoder = nn.DataParallel(momentum_encoder).to(dev)


TrainingDataset = FFHQ(root_dir=config.dataset_root, Resize = Resize, JRC = JRC,
                       transform=Transform, distortion=Distortion, normalize=Normalize,
                       train=True)

dloader = DataLoader(dataset=TrainingDataset, batch_size=config.batch_size, num_workers=32, shuffle=True)
dlen = 69000

# loss history
loss_hist = []

# If resume, load ckpt and loss history
if config.resume:
    file_name = 'ckpt_' + str(config.start_epoch) + '.pkl'
    ckpt = torch.load(os.path.join(weight_path, file_name))
    encoder.load_state_dict(ckpt['encoder'])
    
    try:
        with open(os.path.join(loss_path, 'loss.pkl'), 'rb') as f:
            iter_per_epoch = int(dlen / config.batch_size)
            start_iter = config.start_epoch * iter_per_epoch
            loss_hist = pickle.load(f)[:start_iter]
    except:
        pass
    
for param in momentum_encoder.parameters():
    param.requires_grad = False

# Optimizer
optimizer = optim.SGD(encoder.parameters(), 
                      lr=config.lr, 
                      momentum=config.SGD_momentum, 
                      weight_decay=config.weight_decay)


scheduler1 = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.001, total_iters=5)
scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
                                                       T_max=config.max_epoch)
scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[5])


# Loss function
crossentropy = nn.CrossEntropyLoss()

''' ######################## < Step 3 > Define methods ######################## '''

def momentum_step(m=1):
    '''
    Momentum step (Eq (2)).

    Args:
        - m (float): momentum value. 1) m = 0 -> copy parameter of encoder to key encoder
                                     2) m = 0.999 -> momentum update of key encoder
    '''
    params_q = encoder.state_dict()
    params_k = momentum_encoder.state_dict()
    
    dict_params_k = dict(params_k)
    
    for name in params_q:
        theta_k = dict_params_k[name]
        theta_q = params_q[name].data
        dict_params_k[name].data.copy_(m * theta_k + (1-m) * theta_q)
    
    momentum_encoder.load_state_dict(dict_params_k)

def update_lr(epoch):
    '''
    Learning rate scheduling.

    Args:
        - epoch (float): Set new learning rate by a given epoch.
    '''
    
    if epoch < 120:
        lr = config.lr
    elif 120 <= epoch and epoch < 160:
        lr = config.lr * 0.1
    elif 160 <= epoch:
        lr = config.lr * 0.01
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


''' ######################## < Step 4 > Start training ######################## '''

# Initialize momentum_encoder with parameters of encoder.
momentum_step(m=0)

# Initialize queue.
print('\n[3 / 3]. Initializing a queue with %d keys.' % config.num_keys)
queue = []
i=0
with torch.no_grad():
    for (_, img, _) in tqdm(dloader):
        BN, _, SR,_ = img.shape
        # breakpoint()
        # DifFactor = 0.991**torch.randint(0, 1000, (BN,))[:,None,None,None] #N
        DifFactor = torch.rand(BN)[:,None,None,None]
        Noise = torch.randn_like(img)
        img = torch.sqrt(DifFactor)*img+torch.sqrt(1-DifFactor)*Noise 
        key_feature, _ = momentum_encoder(img.to(dev))
        queue.append(key_feature)

        if i == (config.num_keys / config.batch_size) - 1:
            break
        i+=1
    queue = torch.cat(queue, dim=0)
    
# Training
print('\nStart training!')
epoch = 0 if not config.resume else config.start_epoch
total_iters = 0 if not config.resume else int(dlen / config.batch_size) * config.start_epoch
mse = torch.nn.MSELoss()
AdAvgP = torch.nn.AdaptiveAvgPool2d(output_size=(2,2))

NNS = 10
while(epoch < config.max_epoch):
    print("lr: ",optimizer.param_groups[0]['lr'])
    
    for i, (x_q, x_k, _) in enumerate(dloader):
        BN, _, SR,_ = x_q.shape
        encoder.train()
        momentum_encoder.train()
        encoder.zero_grad()

        # Shffled BN : shuffle x_k before distributing it among GPUs (Section. 3.3)
        if config.shuffle_bn:
            idx = torch.randperm(x_k.size(0))
            x_k = x_k[idx]
            
        # x_q, x_k : (N, 3, 64, 64)    
        Noise = torch.randn_like(x_k)[:,None,:,:,:] # (N, 1, 3, 64, 64)  
#        DifFactor = 0.991**torch.randint(0, 1000, (BN,NNS))[:,:,None,None,None] #N, 100
        DifFactor = torch.rand(BN,NNS)[:,:,None,None,None]
        x_k = x_k[:,None,:,:,:]
        x_k = torch.sqrt(DifFactor)*x_k+torch.sqrt(1-DifFactor)*Noise # N, 100, 3, 64, 64
        x_k = x_k.reshape(NNS*BN, 3, SR, SR)
        x_q, x_k = x_q.to(dev), x_k.to(dev)
        q, colorq = encoder(x_q) # q : (N, 128)
        colorq = Normalize(colorq)
        with torch.no_grad():
            k, _= momentum_encoder(x_k) # k : (N, 128)
            # k = k.detach()
            k = k.reshape(BN, NNS, 128) # 256, 100, 128
        Closs = mse(colorq, AdAvgP(x_q).detach())
        # Shuffled BN : unshuffle k (Section. 3.3)
        if config.shuffle_bn:
            k_temp = torch.zeros_like(k)
            for i, j in enumerate(idx):
                k_temp[j] = k[i]
            k = k_temp

        # Positive sampling q & k

        
        l_pos = torch.sum(q[:,None,:] * k, dim=2, keepdim=True).reshape(-1, 1) # (N, 100, 1)

        # Negative sampling q & queue
        l_neg = torch.mm(q, queue.t()).repeat(NNS, 1) # (N, 4096)

        # Logit and label
        logits = torch.cat([l_pos, l_neg], dim=1) # (N, 4097) witi label [0, 0, ..., 0]
        labels = torch.zeros(logits.size(0), dtype=torch.long).to(dev)

        # Get loss and backprop
        loss = crossentropy(logits / config.temperature, labels)+Closs
        loss.backward()

        # Encoder update
        optimizer.step()

        # Momentum encoder update
        momentum_step(m=config.momentum)

        # Update dictionary
        k = k.reshape(-1, 128)
        queue = torch.cat([k, queue[:queue.size(0) - k.size(0)]], dim=0)
        
        # Print a training status, save a loss value, and plot a loss graph.
        total_iters = total_iters + 1
        print('[Epoch : %d / Total iters : %d] : loss : %f ...' %(epoch, total_iters, loss.item()))
        loss_hist.append(loss.item())
        
        if total_iters % config.plot_iter == 0:
            util.enc_loss_plot(loss_hist, loss_path, record_iter=1)
        
    epoch += 1
    
    # Update learning rate
    scheduler.step()
    # update_lr(epoch)
    
    # Save
    if (epoch - 1) % config.save_weight_epoch == 0:
        path_ckpt = os.path.join(weight_path, 'ckpt_' + str(epoch-1) + '.pkl')
        ckpt = {
            'encoder': encoder.state_dict(),
            'momentum_encoder': momentum_encoder.state_dict()
        }
        torch.save(ckpt, path_ckpt)
        
        with open(os.path.join(loss_path, 'loss.pkl'), 'wb') as f:
            pickle.dump(loss_hist, f)
