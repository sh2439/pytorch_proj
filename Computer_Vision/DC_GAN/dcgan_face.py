import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# import pytorch packages
import torch
import torchvision

from torch.autograd import Variable
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torchvision.utils import make_grid

from tqdm.auto import tqdm

data_root = '/home/sh2439/pytorch_tutorials/gender_detect/combined'
### Set hyperparameters
# check torch device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# number of epochs
num_epochs = 100
# batch size
batch_size = 128
# learning rate
learning_rate = 0.0002
# betas
betas = [0.5,0.999]

def main():
    
    ### 1. Prepare the dataset
    dataset = dsets.ImageFolder(root = data_root,
                            transform = transforms.Compose([
                                transforms.Resize(64),
#                                 transforms.CenterCrop(128),
                                transforms.ToTensor(),
                                transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
                            ])
                           )
    
    ### 2. Make the dataset iterable
    data_loader = torch.utils.data.DataLoader(dataset = dataset,
                                          batch_size = batch_size,
                                        shuffle = True)
    
    ### 3. Instantiate the model class
    Dis = Dis_Net()
    Gen = Gen_Net()
    # put the models to gpu if possible
#     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    Dis.to(device)
    Gen.to(device)
    if torch.cuda.device_count() >1:
        Dis = nn.DataParallel(Dis)
        Gen = nn.DataParallel(Gen)
        torch.backends.cudnn.benchmark = True
    # initialize the model weights
    Dis.apply(init_weights)
    Gen.apply(init_weights)
    
    ### 4. Instantiate loss and optimizer
    # binary cross entropy loss
    criterion = nn.BCELoss()
    
    d_optimizer = torch.optim.Adam(Dis.parameters(), lr = learning_rate, betas = betas)
    g_optimizer = torch.optim.Adam(Gen.parameters(), lr = 0.0004, betas = betas)
    
    ### 5. Train the model
    
    torch.manual_seed(1)
    test_noise = get_noise(16)
    
    for epoch in tqdm(range(num_epochs)):
    
        for batch_idx, (real_images, _) in enumerate(data_loader):

            ### Train the discriminator

            # process real and fake data
            real_images = Variable(real_images).to(device)
            noise = get_noise(real_images.size(0))
            fake_images = Gen(noise)
            # train the discriminator
            loss_D, preds_real, preds_fake = train_dis(Dis, d_optimizer, criterion, real_images, fake_images)


            ### Train the generator
            noise = get_noise(real_images.size(0))
            fake_images = Gen(noise)

            loss_G, preds_fake_g = train_gen(Dis, g_optimizer,criterion, fake_images)

            ### Compute the D(x), D(G(z1)), and D(G(z2))

            D_x = preds_real.mean().cpu()
            D_G_z1 = preds_fake.mean().cpu()
            D_G_z2 = preds_fake_g.mean().cpu()

            if batch_idx % 50 == 0:
                print('epoch:{}, iters:{}, loss_D:{}, loss_G:{}'.format(epoch, batch_idx, loss_D.item(), loss_G.item()))

                print('D(x):{}, D(G(z1)):{}, D(G(z1)):{}'.format(D_x, D_G_z1, D_G_z2))

            # show the test images every 500 iters
#             torch.manual_seed(1)
#             test_noise = get_noise(16)

            if batch_idx % 500 == 0:
                test_imgs = Gen(test_noise).cpu().detach()
#                 print(test_imgs.size())
                show_grid(test_imgs)
    
    
### Helper functions

class Dis_Net(nn.Module):
    """ The discriminator net. The size of input should be n*3*64*64 tensor.
    """
    def __init__(self):
        super(Dis_Net, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(3,64,kernel_size = (4,4), stride = (2,2), padding = (1,1), bias = False),
            nn.LeakyReLU(0.2, inplace = True)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size = (4,4), stride = (2,2), padding = (1,1), bias = False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace = True)
        )
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size = (4,4), stride = (2,2), padding = (1,1), bias = False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace = True)
        )
        
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size= (4,4), stride = (2,2), padding = (1,1), bias = False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace = True)
        
        )
        
        self.layer5 = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size = (4,4), stride = (1,1), bias = False),
            nn.Sigmoid()
        
        )
        
    def forward(self, x):
        """Input: 3*64*64 tensors
        """
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        
        return x
    
class Gen_Net(nn.Module):
    """ The generator net. The size of input should be n*100*1*1 tensor.
    """
    
    def __init__(self):
        super(Gen_Net, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(100, 512, kernel_size = (4,4), stride = (1,1), bias = False),
            nn.BatchNorm2d(512),
            nn.ReLU(True)
        )
        
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size = (4,4), stride = (2,2), padding = (1,1), bias = False),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )
        
        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size = (4,4), stride = (2,2), padding = (1,1), bias = False),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )
        
        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size = (4,4), stride = (2,2), padding = (1,1), bias = False),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )
        self.layer5 = nn.Sequential(
            nn.ConvTranspose2d(64, 3, kernel_size = (4,4), stride = (2,2), padding = (1,1), bias = False),
            nn.Tanh()
        )
        
    def forward(self, x):
        """Input: tensor of length z
        """
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        
        return x
def show(img):
    """Given the tensor of image after the normalization, show the image.
    """
    # inverse normalization
    img = img * 0.5 + 0.5
    plt.figure(figsize = (5,5))
    plt.imshow(img.permute(1,2,0))
    plt.show()
    return

def show_grid(images, title = None, rows = 4):
    """Given the images tensors, show the multiple images
    """
    num = images.size(0)
    
    fig = plt.figure(figsize= (6,6))
    plt.title(title)
    cols = int(num/rows)
    
    for i in range(1, num+1):
        
        
        img = images[i-1]
        img = (img*0.5+0.5).permute(1,2,0)
        fig.add_subplot(rows, cols, i)
        plt.axis('off')
        plt.imshow(img)
    
    plt.show()
    return

def init_weights(m):
    
    if type(m) == nn.Conv2d:
        nn.init.normal_(m.weight.data, 0 , 0.02)
    elif type(m) == nn.BatchNorm2d:
        nn.init.normal_(m.weight.data, 1, 0.02)
        nn.init.constant_(m.bias.data, 0)
        
def get_noise(size):
    """ Generate random noise to feed into the generator. 
    """
    n = torch.randn(size, 100, 1,1, device = device)
    
    return n


def train_dis(model, optimizer, criterion ,real_data, fake_data):
    """ Train the discriminator given the real and fake data.
    """
    # reset gradients
    optimizer.zero_grad()
    
    ### Train on real data
    preds_real = model(real_data)
    real_labels = Variable(torch.ones((real_data.size(0),), device = device))
    loss_real = criterion(preds_real, real_labels)
    
    loss_real.backward()
    
    ### Train on fake data
    preds_fake = model(fake_data)
    fake_labels = Variable(torch.zeros((real_data.size(0),), device = device))
    loss_fake = criterion(preds_fake, fake_labels)
    
    loss_fake.backward()
    
    loss_D = loss_real + loss_fake
    # update
    optimizer.step()
    return loss_D, preds_real, preds_fake

def train_gen(model, optimizer, criterion ,fake_data):
    """ Train the generator given the fake data.
    """
    
    optimizer.zero_grad()
    
    fake_preds_g = model(fake_data)
    real_labels_g = Variable(torch.ones((fake_data.size(0),), device = device))
    # compute the loss
    loss_G = criterion(fake_preds_g, real_labels_g)
    
    loss_G.backward()
    
    # update
    optimizer.step()
    
    return loss_G, fake_preds_g


if __name__ == '__main__':
    main()

    
