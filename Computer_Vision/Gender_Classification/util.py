import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import torch.nn.functional as F

import shutil

import os
import os.path as osp
from PIL import Image

def show(img):
    """Show the image.
    """
    plt.figure(figsize=(5,5))
    plt.imshow(img)
    plt.show()

def get_target(name):
    """Transform the targets to binary labels.
    """
    if name %2 ==0:
        target = 0
    else:
        target = 1
    return target

def preprocess(data):
    """Preprocess the data to the tensor which can be fed to the neural network.
    """
    mean = np.array([129.1863,104.7624,93.5940])/256
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean,std = [1,1,1])   
    ])
    
    data = transform(data)
    #data = data/256
    # RGB to BGR
    data = data[[2,1,0],:,:]
    # switch the height and width
    data = data.permute(0,2,1)

    return data

def count_paras(paras):
    """Count the trainable parameters of the model.
    """
    return sum(p.numel() for p in paras if p.requires_grad)

    

def save_checkpoint(state, is_best, filename = 'checkpoint.pth.tar', folder = ''):
    """ Save the lastest checkpoint and the best model so far.
    """
    # make directory if folder doesn't exist
    if not os.path.exists(folder):
        os.makedirs(folder)
    torch.save(state, folder + '/' + filename)
    
    if is_best:
        shutil.copyfile(folder + '/' + filename, folder + '/model_best.pth.tar')
