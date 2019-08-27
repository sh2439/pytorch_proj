### Import necessary packages
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import os
import os.path as osp

# pytorch packages
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.functional as F
from torchvision.utils import make_grid

from tqdm.auto import tqdm

### Load the dataset
data_root = '/home/sh2439/pytorch_tutorials/pytorch_notebooks/data'
datasets = dsets.MNIST(root = data_root,
                      transform = transforms.Compose([
                          transforms.ToTensor(),
                          transforms.Normalize([0.5],[0.5])
                      ]),
                      train = True)

### Make the dataset iterable
batch_size = 100
data_loader = torch.utils.data.DataLoader(dataset = datasets,
                                         batch_size = batch_size ,
                                         shuffle = True)

num_batches = int(len(datasets)/batch_size)

### Build the model class
# Build the discriminator
class Dis(nn.Module):
    
    def __init__(self):
        super(Dis,self ).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Linear(784, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        
        self.layer2 = nn.Sequential(
            nn.Linear(1024,512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        
        self.layer3 = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        
        self.layer_out = nn.Linear(256,1)
        
    def forward(self, x):
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = F.sigmoid(self.layer_out(x))
        
        return x
    
# Build the generator
class Gen(nn.Module):
    
    def __init__(self):
        super(Gen,self).__init__()
        
        self.num_features = 100
        
        self.layer1 = nn.Sequential(
            nn.Linear(self.num_features, 256),
            nn.LeakyReLU(0.2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Linear(256,512),
            nn.LeakyReLU(0.2)
        )
        
        self.layer3 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2)
        
        )
        
        self.layer_out = nn.Sequential(
            nn.Linear(1024, 784),
            nn.Tanh()
        )
        
    def forward(self, x):

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer_out(x)

        return x
    
### Instantiate the model class and put the model to GPU
Dis_model = Dis()
Gen_model = Gen()

### Put the model to the gpu if possible
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if torch.cuda.device_count() > 1:
    Dis_model = nn.DataParallel(Dis_model).cuda()
    Gen_model = nn.DataParallel(Gen_model).cuda()
    
    torch.backends.cudnn.benchmark = True
    
Dis_model.to(device)
Gen_model.to(device)
print('optimizer', flush = True)
### Loss and optimizer
dis_optimizer = torch.optim.Adam(Dis_model.parameters(), lr = 0.0002)
gen_optimizer = torch.optim.Adam(Gen_model.parameters(), lr = 0.0002)

criterion = nn.BCELoss()

### Helper functions
def get_real_label(real_data):
    """Generate real data labels, all ones
    """
    num = real_data.size(0)
    labels = Variable(torch.ones(num,1)).to(device)
    
    return labels

def get_fake_label(fake_data):
    """Generate fake data labels, all zeros.
    """
    num = fake_data.size(0)
    labels = Variable(torch.zeros(num,1)).to(device)
    
    return labels

def train_dis(model,optimizer, real_data, fake_data):
    """ Train the discriminator given the true data and the fake data.
    """
    optimizer.zero_grad()
    
    # train on real data
    real_preds = model(real_data)
    loss_real = criterion(real_preds, get_real_label(real_data))
    loss_real.backward()
    
    # train on fake data
    fake_preds = model(fake_data)
    loss_fake = criterion(fake_preds, get_fake_label(fake_data))
    loss_fake.backward()
    
    loss_D = loss_real + loss_fake
    # update
    optimizer.step()
    
    return loss_D, real_preds, fake_preds
    
def train_gen(model, optimizer, fake_data):
    """Helper function to train the generator.
    """
    # reset optimizer
    optimizer.zero_grad()
    
    # sample noise and generate fake_data
    preds = model(fake_data)
    
    loss = criterion(preds, get_real_label(preds))
    loss.backward()

    optimizer.step()
    return loss

def get_noise(size):
    """generate `size` number of random images
    """
    n = Variable(torch.randn(size, 100)).to(device)
    return n

def show_grid(images, title = None, rows = 4):
    """Given the images tensors, show the multiple images in a grid.
    """
    num = images.size(0)
    
    fig = plt.figure(figsize= (5,5))
    plt.title(title)
    cols = int(num/rows)
    
    for i in range(1, num+1):
        img = images[i-1].view(28,28)
        fig.add_subplot(rows, cols, i)
        plt.axis('off')
        plt.imshow(img, cmap = 'gray')
    
    plt.show()
    return


test_input = get_noise(16)

num_epochs = 200

for epoch in tqdm(range(num_epochs)):
    
    for batch_idx, (real_images, _) in enumerate(data_loader):
        
        ### 1. Train the discriminator
        
        real_images = Variable(real_images.view(-1,28*28)).to(device)
        
        # generate the fake data
        fake_images = Gen_model(get_noise(real_images.size(0))).detach()
        
        # train the discriminator
        loss_D, real_preds, fake_preds = train_dis(Dis_model, dis_optimizer, real_images, fake_images)
        
        ### 2. Train the generator
        
        fake_images = Gen_model(get_noise(real_images.size(0)))
        # train the generator
        loss_G = train_gen(Dis_model, gen_optimizer, fake_images)
    
    # show the loss every epoch
    print('epoch:{}, iters:{}, loss_D:{}, loss_G:{}'.format(epoch, batch_idx, loss_D.item(), loss_G.item()))

    # show the test images every 1 epochs
    if (epoch % 20) ==0:
        test_imgs = Gen_model(test_input).cpu().detach()
        show_grid(test_imgs)
    


