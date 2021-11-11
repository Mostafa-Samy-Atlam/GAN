# -*- coding: utf-8 -*-
"""
@author: Eng. Mostafa
"""
#Importing Libraries:

import matplotlib.pyplot as plt
import itertools
import torch.optim as optim
import torchvision.datasets as dset
from torch.utils.data.dataset import Dataset
from IPython.display import HTML
import torchvision
import torchvision.transforms as transforms
import torch
from torch.optim.optimizer import Optimizer, required
from torch.autograd import Variable
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from torch.nn import Parameter

##############################################################################
#Building Spectral Normalization Class:

def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)

class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False


    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)


    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)
##############################################################################
#Setting Batch Size and Number of Epochs: 

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
#Building Discriminator Class using Convolutional Layers and Spectral Normalization:    

class discriminator_model(nn.Module):

  def __init__(self):
    super(discriminator_model, self).__init__()
    self.conv1 = SpectralNorm(nn.Conv2d(1, 128, 4, 2, 1))
    self.conv2 = SpectralNorm(nn.Conv2d(128, 256, 4, 2, 1))
    self.conv3 = SpectralNorm(nn.Conv2d(256, 512, 4, 2, 1))
    self.conv4 = SpectralNorm(nn.Conv2d(512, 1024, 4, 2, 1))
    self.conv5 = SpectralNorm(nn.Conv2d(1024, 1, 4, 1, 0))
    
  def weight_init(self):
    for m in self._modules:
      normal_init(self._modules[m])
      
  def forward(self, input):
    x = F.leaky_relu(self.conv1(input), 0.2)
    x = F.leaky_relu(self.conv2(self.conv2(x)), 0.2)
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
    self.deconv1 = SpectralNorm(nn.ConvTranspose2d(100, 1024, 4, 1, 0))
    self.deconv2 = SpectralNorm(nn.ConvTranspose2d(1024, 512, 4, 2, 1))
    self.deconv3 = SpectralNorm(nn.ConvTranspose2d(512, 256, 4, 2, 1))
    self.deconv4 = SpectralNorm(nn.ConvTranspose2d(256, 128, 4, 2, 1))
    self.deconv5 = nn.ConvTranspose2d(128, 1, 4, 2, 1)
    
  def weight_init(self):
    for m in self._modules:
      normal_init(self._modules[m])
      
  def forward(self, input):
    x = F.relu(self.deconv1(self.deconv1(input)))
    x = F.relu(self.deconv2(self.deconv2(x)))
    x = F.relu(self.deconv3(self.deconv3(x)))
    x = F.relu(self.deconv4(self.deconv4(x)))
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

#End
##############################################################################