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
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 1
n_epochs = 10

# Optimizers
lr=0.0002
b1=0.5
b2=0.999

# number of images used for training
N = 3

# loss weights
alpha = 10      # L1 loss
beta = 0.0001   # perceptual loss

# number of threads for loading dataset
workers = 2

# whether to reorder the dataset
shuffle = False

image_size = 256
img_shape = (image_size, image_size, 3)

# file path
watermark_path = "dataset/train/watermarked/"
original_path = "dataset/train/original/"
result_path = "result"

# define transformers
transform = transforms.Compose([transforms.Resize(image_size), 
                                       transforms.CenterCrop(image_size),  
                                       transforms.ToTensor(), 
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
img_transform = transforms.Compose([transforms.Resize(image_size), transforms.CenterCrop(image_size)])

# -------- define dataset -------- #
class MyDataset(Dataset):
    def __init__(self, transform = None):
        imgs = []
        # Please name your images in the dataset as "1.png" "2.png" 
        for i in range(1,N+1):
            imgs.append((watermark_path+str(i)+"_random.png", original_path+str(i)+".png"))
        
        # Otherwise, you could rewrite this function
                
        self.imgs = imgs
        self.transform = transform
 
    # define return values
    def __getitem__(self, index):
        fn, label = self.imgs[index]   # fn: path for watermarked images; label: ground truths(original images)
        img = Image.open(fn).convert('RGB')
        tar = Image.open(label).convert('RGB')
        
        if self.transform is not None:
            #print(img.type())
            img = self.transform(img)
            tar = self.transform(tar)
                    
        return fn, label, img, tar
 
    def __len__(self):
        return len(self.imgs)
    
# -------- define dataloader -------- #
train_set = MyDataset(transform)
dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=shuffle, num_workers=workers)

# -------- define transform converter -------- #
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

# -------- define networks -------- #
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


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv0 = nn.Conv2d(6, 64 ,3, stride=2, padding=1)
        self.norm1 = nn.InstanceNorm2d(64)
        self.conv1 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)
        self.norm2 = nn.InstanceNorm2d(128)
        self.conv2 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)
        self.norm3 = nn.InstanceNorm2d(256)
        self.conv3 = nn.Conv2d(256, 1, 3, stride=1, padding=1)
        
    def forward(self, x):
        x = self.conv0(x)
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.norm2(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.norm3(x)
        x = self.conv3(x)
        x = torch.nn.Sigmoid()(x.view(x.size()[0],-1).mean(1))
        return x    

# -------- define perceptual loss ---------- #
# change this if your image size is not 256*256
# whc stands for width, height and number of channels of the output of relu2_2
whc = 128*128*128 

class VGG16(torch.nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        model = models.vgg16(pretrained=False)
        pre = torch.load('vgg16-397923af.pth')
        model.load_state_dict(pre)
        vgg_pretrained_features = model.features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()

        # Slice 1 -> layers 1-4 of VGG
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])

        # Slice 2 -> layers 4-9 of VGG
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])

        # Slice 3 -> layers 9-16 of VGG
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])

        # Slice 4 -> layers 16-23 of VGG
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        out = self.slice1(x)
        relu1_2 = out  # Snapshot output of relu1_2
        out = self.slice2(relu1_2)
        relu2_2 = out
        out = self.slice3(relu2_2)
        relu3_3 = out
        out = self.slice4(relu3_3)
        relu4_3 = out

        output_tuple = namedtuple("VGGOutputs", ['relu1_1', 'relu2_2', 'relu3_3', 'relu4_3'])
        out = output_tuple(relu1_2, relu2_2, relu3_3, relu4_3)
        return out

vggmodel = VGG16().to(device)

def VGGls(gxs, ys):
    gx = gxs.clone().detach()
    y = ys.clone().detach()

    features_gx = vggmodel.forward(gx)
    features_y = vggmodel.forward(y)
    t = features_gx.relu2_2-features_y.relu2_2
    t = torch.norm(t)
    pl = (t * t / whc) / batch_size
    return pl 

# -------- initialize loss, optimizers and networks -------- #
# Loss function
adv_loss = torch.nn.BCELoss()
l1_loss = torch.nn.L1Loss()
per_loss = 0.0

# Initialize generator and discriminator
G_net = Generator().to(device)
D_net = Discriminator().to(device)

# Optimizers
optimizer_G = torch.optim.Adam(G_net.parameters(), lr=lr, betas=(b1, b2))
optimizer_D = torch.optim.Adam(D_net.parameters(), lr=lr, betas=(b1, b2))

G_net.train()
D_net.train()

# -------- start to train the networks -------- #
for epoch in range(0,n_epochs):
    
    for i, (x,y,img,tar) in enumerate(dataloader):
        
        ones = torch.ones(batch_size).to(device)
        zeros = torch.zeros(batch_size).to(device)

        img = img.to(device)
        tar = tar.to(device)
        
        # -------- train D net -------- #
        optimizer_D.zero_grad()
        gen_tar = G_net(img).detach().to(device)
        real_x = torch.cat((img, tar), dim=1).to(device)
        fake_x = torch.cat((img, gen_tar), dim=1).to(device)
        real_loss = adv_loss(D_net(real_x), ones)
        fake_loss = adv_loss(D_net(fake_x), zeros)
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()
        
        # -------- train G net -------- #
        optimizer_G.zero_grad()
        gen_tar = G_net(img)
        fake_x = torch.cat((img, gen_tar), dim=1)
        L1_loss = l1_loss(tar, gen_tar)
        per_loss = VGGls(gen_tar,tar)
        g_loss = adv_loss(D_net(fake_x), ones) + alpha*L1_loss + beta*per_loss

        g_loss.backward()
        optimizer_G.step()

        print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [L1 loss: %f] [per loss: %f]" % (epoch, n_epochs, i, len(dataloader), d_loss.item(), g_loss.item(), L1_loss, per_loss), )
    
    # output results for every 10 epochs
    if (epoch+1)%5 == 0:
        plt_size = 1 #plt_size<batch_size, <5 recommended
        plt.figure(figsize=(plt_size*10,30))
        for i in range(plt_size):
            ax = plt.subplot(3,plt_size,i+1)
            plt.imshow(transform_convert(img[i], transform))
            ax = plt.subplot(3,plt_size,plt_size+i+1)
            plt.imshow(transform_convert(tar[i], transform))
            ax = plt.subplot(3,plt_size,plt_size*2+i+1)
            plt.imshow(transform_convert(gen_tar[i], transform))
            plt.savefig(result_path+"/img_"+str(epoch)+".png")
        # save parameters of the models
        stateG = {'model': G_net.state_dict(), 'optimizer': optimizer_G.state_dict(), 'epoch': epoch}
        torch.save(stateG, result_path+'/Gnet_'+str(epoch)+".pth")

        stateD = {'model': D_net.state_dict(), 'optimizer': optimizer_D.state_dict(), 'epoch': epoch}
        torch.save(stateD, result_path+'/Dnet_'+str(epoch)+".pth")

print("Done! batch_size="+str(batch_size)+", N=" + str(N) +", results saved in folder:"+result_path)