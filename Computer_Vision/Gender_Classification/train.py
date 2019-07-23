import sys
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import torch.nn.functional as F
from tqdm.auto import tqdm
import shutil


import os
import os.path as osp
from PIL import Image
import matplotlib.pyplot as plt

import configargparse
import argparse

from util import *
from vgg_16 import *


parser = configargparse.ArgParser(description='PSMNet')
parser.add('-c', '--config', required=True,
           is_config_file=True, help='config file')

parser.add_argument('--pretrain', type = int, default =0,
                   help = 'pretrain or not')
parser.add_argument('--load_weights',
                   help = 'Load pretrained VGG model or weights')
parser.add_argument('--datapath',
                   help = 'root folder that contains gender dataset')
parser.add_argument('--savepath',
                   help = 'folder that contains saved model')
parser.add_argument('--lr', type = float, default = 0.001,
                   help = 'learning rate')
parser.add_argument('--batch_size', type = int, default = 64,
                   help = 'batch size')
parser.add_argument('--epochs', type = int, default = 30,
                   help = 'number of epochs')
parser.add_argument('--freeze',type = int, default = 0,
                   help = 'Freeze the parameters in the conv layers')

args = parser.parse_args()


### 1. Load the dataset


data_path = args.datapath
trainval_dataset = dsets.ImageFolder(root = data_path,
                                    target_transform = get_target,
                                    transform = preprocess)

### train-val split
torch.manual_seed(0)

train_size = int(len(trainval_dataset)*0.8)
val_size = len(trainval_dataset) - train_size

train_dataset, val_dataset = torch.utils.data.random_split(trainval_dataset,[train_size,val_size])

### Make the dataset iterable
train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                          batch_size = args.batch_size,
                                          shuffle = True)

val_loader = torch.utils.data.DataLoader(dataset = val_dataset,
                                        batch_size = 16,
                                        shuffle = False)

### Build the model class

        
### Instantiate the model class

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = vgg_16()

if torch.cuda.device_count() > 1:
    
    print("Let's use", torch.cuda.device_count(), "GPUs!",flush = True)
    model = nn.DataParallel(model).cuda()
    torch.backends.cudnn.benchmark = True

model.to(device)



### Load the weights
if args.pretrain == 1 :
    print('load weights...')
    weights = np.load(args.load_weights, encoding = 'latin1').item()
    weights_new = {}

    for key in weights.keys():
        if 'fc' in key:
            continue
        weights_new[key+'.'+'weight'] = torch.tensor(weights[key]['weights']).permute(3,2,0,1)
        weights_new[key+'.'+'bias'] = torch.tensor(weights[key]['biases'])

    state = model.module.state_dict()
    state.update(weights_new)
    model.module.load_state_dict(state)

    if args.freeze == 1:
        print('finetune the model', flush = True)

        for name,para in model.named_parameters():
            if 'fc' not in name:
                para.requires_grad = False
        trainable_paras = []
        for para in model.parameters():
            if para.requires_grad is True:
                trainable_paras.append(para)
        print('length of paras:',len(trainable_paras), flush = True)

    else:
        print('Train all the layers', flush = True)
        trainable_paras = list(model.parameters())
    print('number of parameters', count_paras(trainable_paras), flush = True)
    
else:
    trainable_paras = list(model.parameters())
    print('train from scratch: ', count_paras(trainable_paras), flush = True)

### Loss and optimizer
criterion = nn.CrossEntropyLoss()

learning_rate = args.lr
optimizer = torch.optim.Adam(trainable_paras, lr = learning_rate)
print('start training...', flush =True)

### Train the model
num_epochs = args.epochs
iter_count = 0
best_accuracy = 0
for epoch in tqdm(range(num_epochs)):
    epoch += 1
    
    for images, labels in train_loader:
        images = Variable(images).to(device)
        labels = Variable(labels).to(device)
        #print(torch.max(labels))
      
        optimizer.zero_grad()
        # set the model to training mode
        model.train()
        outputs = model(images)
        
        loss = criterion(outputs, labels)
        loss.backward()
        
        optimizer.step()
        iter_count+=1
        
        if iter_count % 16 ==0:
            print('epoch {}, iter {}, loss {}'.format(epoch, iter_count, loss.item()), flush = True)
        
    # Evaluation
    total = 0
    correct = 0
    for images,labels in val_loader:
        model.eval()
        
        with torch.no_grad():
            images = Variable(images).to(device)
            labels = Variable(labels).to(device)
            
            outputs = model(images)
            _,preds = torch.max(outputs, 1)
            
            correct += (preds.cpu() == labels.cpu()).sum()
            total += labels.size(0)
    accuracy = correct.item()*100/total
    
    is_best = accuracy > best_accuracy
    best_accuracy = max(accuracy, best_accuracy)
    
    print('epoch {}, loss {}, accuracy {}'.format(epoch, loss.item(), accuracy),flush= True)
    # save the model
    save_checkpoint({'epoch':epoch,
                    'state_dict': model.state_dict(),
                    'best_accuracy': best_accuracy},
                   is_best, folder = args.savepath)
