# -*- coding: utf-8 -*-
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import torch.utils.data
from PIL import Image
import matplotlib
matplotlib.use('Agg') #use non-interactive backend, or error
import matplotlib.pyplot as plt

import argparse
import os
import numpy as np
import math

from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

from torchvision import models
from collections import namedtuple

# -------- set parameters -------- #
model_path="result/Gnet_4.pth"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 1
n_epochs = 1
N = 2 # number of images for test
workers = 2
shuffle = False
lr=0.0002
b1=0.5
b2=0.999
# 图片大小，tensor不允许图像大小不一致，所以还是把每张图都裁成方的
image_size = 256
img_shape = (image_size, image_size, 3)

# 水印图和原图文件夹路径
original_path = "dataset/test/original/"
watermark_path = "dataset/test/watermarked/"

result_path = "test_result/"
transform = transforms.Compose([transforms.Resize(image_size), 
                                       transforms.CenterCrop(image_size),  
                                       #transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(), 
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
img_transform = transforms.Compose([transforms.Resize(image_size), transforms.CenterCrop(image_size)])

# -------- define dataset -------- #
class MyDataset(Dataset):
    def __init__(self, transform = None):
        imgs = []
        for i in range(1,N+1):
            imgs.append((watermark_path+str(i)+"_random.png", original_path+str(i)+".png"))
        self.imgs = imgs
        self.transform = transform
 
    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = Image.open(fn).convert('RGB')
        tar = Image.open(label).convert('RGB')
        
        if self.transform is not None:
            img = self.transform(img) 
            tar = self.transform(tar)
        
        return fn, label, img, tar
 
    def __len__(self):
        return len(self.imgs)
    
# -------- define dataloader -------- #
train_set = MyDataset(transform)
dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=shuffle, num_workers=workers)

def transform_convert(img_tensor, transform):
    """
    param img_tensor: tensor
    param transforms: torchvision.transforms
    """
    img_tensor = img_tensor.cpu()
    if 'Normalize' in str(transform):
        normal_transform = list(filter(lambda x: isinstance(x, transforms.Normalize), transform.transforms))
        mean = torch.tensor(normal_transform[0].mean, dtype=img_tensor.dtype, device=img_tensor.device)
        std = torch.tensor(normal_transform[0].std, dtype=img_tensor.dtype, device=img_tensor.device)
        img_tensor.mul_(std[:,None,None]).add_(mean[:,None,None])
        
    img_tensor = img_tensor.transpose(0,2).transpose(0,1)  # C x H x W  ---> H x W x C
    
    if 'ToTensor' in str(transform) or img_tensor.max() < 1:
        img_tensor = img_tensor.detach().numpy()*255
    
    if isinstance(img_tensor, torch.Tensor):
    	img_tensor = img_tensor.numpy()
    
    if img_tensor.shape[2] == 3:
        img = Image.fromarray(img_tensor.astype('uint8')).convert('RGB')
    elif img_tensor.shape[2] == 1:
        img = Image.fromarray(img_tensor.astype('uint8')).squeeze()
    else:
        raise Exception("Invalid img shape, expected 1 or 3 in axis 2, but got {}!".format(img_tensor.shape[2]))
        
    return img

# -------- define network -------- #
def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)
        
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.conv0 = nn.Conv2d(3, 64, 3, stride=2, padding=1)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv1 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.norm2 = nn.InstanceNorm2d(128)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.norm3 = nn.InstanceNorm2d(256)
        self.relu3 = nn.LeakyReLU(0.2, inplace=True)
        self.conv3 = nn.Conv2d(256, 256, 3, stride=2, padding=1)
        self.relu4 = nn.ReLU()
        self.upsm4 = nn.Upsample(scale_factor=2)
        self.conv4 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.norm4 = nn.InstanceNorm2d(256)
        self.relu5 = nn.ReLU()
        self.upsm5 = nn.Upsample(scale_factor=2)
        self.conv5 = nn.Conv2d(512, 128, 3, stride=1, padding=1)
        self.norm5 = nn.InstanceNorm2d(128)
        self.relu6 = nn.ReLU()
        self.upsm6 = nn.Upsample(scale_factor=2)
        self.conv6 = nn.Conv2d(256, 64, 3, stride=1, padding=1)
        self.norm6 = nn.InstanceNorm2d(64)
        self.relu7 = nn.ReLU()
        self.upsm7 = nn.Upsample(scale_factor=2)
        self.conv7 = nn.Conv2d(128, 3, 3, stride=1, padding=1)
        
        self.apply(init_weights)
       

    def forward(self, x):
        x1 = self.conv0(x)
        x1 = self.relu1(x1)
        x2 = self.conv1(x1)
        x2 = self.norm2(x2)
        x3 = self.relu2(x2)
        x3 = self.conv2(x3)
        x3 = self.norm3(x3)
        x4 = self.relu3(x3)
        x4 = self.conv3(x4)
        x4 = self.relu4(x4)
        x4 = self.upsm4(x4)
        x4 = self.conv4(x4)
        x4 = self.norm4(x4)
        x5 = self.relu5(torch.cat((x4, x3), dim=1))
        x5 = self.upsm5(x5)
        x5 = self.conv5(x5)
        x5 = self.norm5(x5)
        x6 = self.relu6(torch.cat((x5, x2), dim=1))
        x6 = self.upsm6(x6)
        x6 = self.conv6(x6)
        x6 = self.norm6(x6)
        x7 = self.relu7(torch.cat((x6, x1), dim=1))
        x7 = self.upsm7(x7)
        out = self.conv7(x7)
        
        return out

G_net = Generator().to(device)
optimizer_G = torch.optim.Adam(G_net.parameters(), lr=lr, betas=(b1, b2))
checkpoint = torch.load(model_path)
G_net.load_state_dict(checkpoint['model'])
optimizer_G.load_state_dict(checkpoint['optimizer'])
epoch = checkpoint['epoch']
G_net.eval()

for i, (x,y,img,tar) in enumerate(dataloader):
    img = img.to(device)
    gen_tar = G_net(img)
    print(i)
    # print(img.size())
    plt.figure(figsize=(10,30))
    ax = plt.subplot(3,1,1)
    plt.imshow(transform_convert(img[0], transform))
    ax = plt.subplot(3,1,2)
    plt.imshow(transform_convert(tar[0], transform))
    ax = plt.subplot(3,1,3)
    plt.imshow(transform_convert(gen_tar[0], transform))
    plt.savefig(result_path+"img_"+str(i)+".png")

print("Done! test result saved in folder:"+result_path)
