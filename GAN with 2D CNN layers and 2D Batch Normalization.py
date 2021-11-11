# -*- coding: utf-8 -*-
"""
@author: Eng. Mostafa
"""
#Importing Libraries:

import matplotlib.pyplot as plt
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as dset
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset
from IPython.display import HTML
import torchvision
import torchvision.transforms as transforms

##############################################################################
#Setting Number of Epochs and Batch Size:

img_size = 64
n_epochs = 10
batch_size = 64
##############################################################################
#Downloading and Reading Dataset:

transform = transforms.Compose([
    transforms.Scale(img_size),
    transforms.ToTensor(),
])

dataset = torchvision.datasets.MNIST('./data/', download=True, transform=transform, train=True)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True)

##############################################################################
#Building Discriminator Class using Convolutional Layers and Batch Normalization:    

class discriminator_model(nn.Module):

  def __init__(self):
    super(discriminator_model, self).__init__()
    self.conv1 = nn.Conv2d(1, 128, 4, 2, 1)
    self.conv2 = nn.Conv2d(128, 256, 4, 2, 1)
    self.conv2_bn = nn.BatchNorm2d(256)
    self.conv3 = nn.Conv2d(256, 512, 4, 2, 1)
    self.conv3_bn = nn.BatchNorm2d(512)
    self.conv4 = nn.Conv2d(512, 1024, 4, 2, 1)
    self.conv4_bn = nn.BatchNorm2d(1024)
    self.conv5 = nn.Conv2d(1024, 1, 4, 1, 0)
    
  def weight_init(self):
    for m in self._modules:
      normal_init(self._modules[m])
      
  def forward(self, input):
    x = F.leaky_relu(self.conv1(input), 0.2)
    x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
    x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
    x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
    x = F.sigmoid(self.conv5(x))
    return x

def normal_init(m):
  if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
    m.weight.data.normal_(0.0, 0.02)
    m.bias.data.zero_()
##############################################################################
#Building Generator Class:    

class generator_model(nn.Module):

  def __init__(self):
    super(generator_model, self).__init__()
    self.deconv1 = nn.ConvTranspose2d(100, 1024, 4, 1, 0)
    self.deconv1_bn = nn.BatchNorm2d(1024)
    self.deconv2 = nn.ConvTranspose2d(1024, 512, 4, 2, 1)
    self.deconv2_bn = nn.BatchNorm2d(512)
    self.deconv3 = nn.ConvTranspose2d(512, 256, 4, 2, 1)
    self.deconv3_bn = nn.BatchNorm2d(256)
    self.deconv4 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
    self.deconv4_bn = nn.BatchNorm2d(128)
    self.deconv5 = nn.ConvTranspose2d(128, 1, 4, 2, 1)
    
  def weight_init(self):
    for m in self._modules:
      normal_init(self._modules[m])
      
  def forward(self, input):
    x = F.relu(self.deconv1_bn(self.deconv1(input)))
    x = F.relu(self.deconv2_bn(self.deconv2(x)))
    x = F.relu(self.deconv3_bn(self.deconv3(x)))
    x = F.relu(self.deconv4_bn(self.deconv4(x)))
    x = F.tanh(self.deconv5(x))
    return x

##############################################################################

generator = generator_model()
discriminator = discriminator_model()
generator.weight_init()
discriminator.weight_init()

generator.cuda()
discriminator.cuda()
BCE_loss = nn.BCELoss()

##############################################################################
# Training the network:
    
for epoch in range(n_epochs):

  D_losses = []
  G_losses = []
  
  for X, _ in train_loader:
    discriminator.zero_grad()
    mini_batch = X.size()[0]
    
    y_real_ = torch.ones(mini_batch)
    y_fake_ = torch.zeros(mini_batch)
    
    X = Variable(X.cuda())
    y_real_ = Variable(y_real_.cuda())
    y_fake_ = Variable(y_fake_.cuda())
    
    D_result = discriminator(X).squeeze()
    D_real_loss = BCE_loss(D_result, y_real_)
    
    z_ = torch.randn((mini_batch, 100)).view(-1, 100, 1, 1)
    z_ = Variable(z_.cuda())
    G_result = generator(z_)
    
    D_result = discriminator(G_result).squeeze()
    D_fake_loss = BCE_loss(D_result, y_fake_)
    D_fake_score = D_result.data.mean()
    D_train_loss = D_real_loss + D_fake_loss
    
    D_train_loss.backward()
    D_losses.append(D_train_loss)
    
    generator.zero_grad()
    
    z_ = torch.randn((mini_batch, 100)).view(-1, 100, 1, 1)
    z_ = Variable(z_.cuda())
    
    G_result = generator(z_)
    D_result = discriminator(G_result).squeeze()
    G_train_loss = BCE_loss(D_result, y_real_)
    G_train_loss.backward()
    G_losses.append(G_train_loss)
    
  print('Epoch {} - loss_d: {:.3f}, loss_g: {:.3f}'.format((epoch + 1),
                                                           torch.mean(torch.FloatTensor(D_losses)),
                                                           torch.mean(torch.FloatTensor(G_losses))))
# End
##############################################################################