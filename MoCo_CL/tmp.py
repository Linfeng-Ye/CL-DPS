from FFHQ_Dataset import FFHQ
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from MotionBlur import RandomMotionBlur
from torchvision import transforms, utils
from tqdm import tqdm
import matplotlib.pyplot as plt

Transform = transforms.Compose([transforms.Resize(256),
                                transforms.RandomResizedCrop((64,64), scale=(0.045, 0.08), 
                                                             ratio=(0.7,1.3)),
                                #transforms.RandomHorizontalFlip(p=0.5),
                                ])
Distortion = nn.Identity()

Normalize = nn.Identity()


TrainingDataset = FFHQ(root_dir="/home/multicompc15/Documents/CppDiffusion/FFHQ256", 
                       transform=Transform, distortion=Distortion, normalize=Normalize,
                       train=True)
dloader = DataLoader(dataset=TrainingDataset, batch_size=1, num_workers=36, shuffle=False)
dlen = 60000
for sample in tqdm(dloader):
    #breakpoint()
    plt.imshow(sample[0][0].transpose(0,2))
    plt.show()
    pass
