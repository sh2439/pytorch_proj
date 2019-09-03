import numpy as np
import pandas as pd
import os
import os.path as osp

# import pytorch packages
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.utils import data

from tqdm.auto import tqdm
import emoji

from util import *



def main():
    
    # Define data root
    data_root = 'data'
    train_name = 'train_emoji.csv'
    test_name = 'tess.csv'
    
    train_dataset = emoji_dataset(data_root, train_name)
    test_dataset = emoji_dataset(data_root, test_name)
    
    ### Define the dataloader
    batch_size = 32
    train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                              shuffle = True,
                                               batch_size = batch_size,
                                              )

    test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                              shuffle = False,
                                               batch_size = 8,
                                              )
    ### Read the GloVe file
    name = '/home/sh2439/pytorch_tutorials/Sequence Model/Week 2/Word Vector Representation/glove.6B.50d.txt'
    words, word2vec_map, word2index, index2word = read_glove(name)
    
    ### Create the embedding weights
    emb_weights = torch.zeros(len(word2vec_map)+1, 50)
    for word, idx in word2index.items():
    #     print(word)
        emb_weights[idx,:] = torch.tensor(word2vec_map[word])
        
        
    device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')

    ### Instantiate the model
    layer_num = 2
    input_dim = 50
    hidden_dim = 128
    output_dim = 5


    emoji_net = Emoji_Net(layer_num, input_dim,hidden_dim, output_dim, emb_weights)

    emoji_net.to(device)
    
    
    ### Loss and optimizer
    criterion = nn.CrossEntropyLoss()

    learning_rate = 0.001
    optimizer = torch.optim.Adam(emoji_net.parameters(), lr = learning_rate)
    
    
    
    ### Start training
    num_epochs = 200

    
    is_best = False
    best_accuracy = 0

    for epoch in tqdm(range(int(num_epochs))):
        emoji_net.train()
        for batch_idx, (inputs, labels) in enumerate(train_loader):


            # clear grads
            optimizer.zero_grad()

            inputs = to_index(inputs,word2index,max_length = 15)
            inputs = Variable(inputs).to(device)

            labels = Variable(labels).to(device)

            # forward pass
            outputs = emoji_net(inputs)

            # get loss
            loss = criterion(outputs, labels)
            # backward
            loss.backward()

            optimizer.step()

            if batch_idx % 5 == 0:
                print('epoch: {}, iters: {}, loss: {}'.format(epoch, 
                batch_idx + epoch*np.ceil(len(train_dataset)/32), loss.item()))


        correct = 0
        total = 0

        with torch.no_grad():
            emoji_net.eval()
            for batch_idx, (inputs, labels) in enumerate(test_loader):

                inputs = to_index(inputs,word2index, max_length = 15)
                inputs = Variable(inputs).to(device)
                labels = Variable(labels)

                outputs = emoji_net(inputs)
                _,preds = torch.max(outputs.data, dim=1)


                total += labels.size(0)
                correct += float((preds.cpu() == labels).sum())

            accuracy = 100* correct/total
            print( 'Epoch: {}, Test Accuracy:{}'.format(epoch, accuracy))

            if accuracy > best_accuracy:
                is_best = True
                best_accuracy = accuracy
                #save_best(is_best, best_accuracy, emoji_net, epoch, 'models/')
            else:
                is_best = False


    
    
    
    
    
    
    
if __name__ == '__main__':
    main()
