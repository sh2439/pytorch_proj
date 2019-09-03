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

device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')


def to_emoji(emoji_dict, label):
    """ Cast a numerical label to the emoji
    """
    
    emoji_new = emoji.emojize(emoji_dict[label],use_aliases=True)
    
    return emoji_new

emoji_dictionary = {0: "\u2764\uFE0F",    # :heart: prints a black instead of red heart depending on the font
                    1: ":baseball:",
                    2: ":smile:",
                    3: ":disappointed:",
                    4: ":fork_and_knife:"}


class emoji_dataset(data.Dataset):
    
    def __init__(self, data_root, data_name):
        """
        """
        
        self.dataset = pd.read_csv(osp.join(data_root, data_name), header = None)
        self.length = len(self.dataset)
        self.data = self.dataset[0]
        self.labels = self.dataset[1]
        
    def __len__(self):
        return self.length
    
    def data(self,index):
        """return the data.
        """
        return self.data[index]
    
    def label(self,index):
        """return the labels of dataset.
        """
        return self.labels[index]
    
    def __getitem__(self, index):
        
        X = self.data[index]
        y = self.labels[index]
        
        return X, y
    
def read_glove(name):
    """Given the path/name of the glove file, return the words(set) and word2vec_map(a python dict)
    """
    file = open(name, 'r')
    # Create set for words and a dictionary for words and their corresponding  
    words = set()
    word2vec_map = {}
    
    data = file.readlines()
    for line in data:
        # add word to the words set.
        word = line.split()[0]
        words.add(word)
        
        word2vec_map[word] = np.array(line.split()[1:], dtype = np.float64)
        
    i = 1
    word2index = {}
    index2word = {}
    for word in words:
        word2index[word] = i
        index2word[i] = word
        i = i+1
        
    return words, word2vec_map, word2index, index2word

def to_index(sentences, word2index, max_length):
    """ Given the word2index dict, maximum length, and inputs, return the numerical inputs
    """
    num = len(sentences)
    out = torch.zeros(num, max_length).long()
    
    for idx, sen in enumerate(sentences):
        
        sen = sen.lower().split()
        
        j = 0
        
        for word in sen:
            word_idx = word2index[word]
            out[idx, j] = word_idx
            j += 1
            
            if j >= max_length:
                break
            
    return out


class Emoji_Net(nn.Module):
    """ The emoji net uses embedding layer, lstm layer and fully-connected layer.
    """
    def __init__(self,layer_num,input_dim, hidden_dim, output_dim, weights):
        super(Emoji_Net, self).__init__()
        self.input_dim = input_dim
        self.layer_num = layer_num
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        
        # the embedding layer
        weights = weights.to(device)
        self.embedding = nn.Embedding.from_pretrained(weights)
        
        # the lstm layer
        self.lstm = nn.LSTM(input_size = self.input_dim, hidden_size = self.hidden_dim, 
                            num_layers = self.layer_num, batch_first = True, dropout = 0.8,bidirectional = True)
        self.dropout = nn.Dropout(0.6)
        
        # the output layer
        self.fc = nn.Linear(hidden_dim*2, output_dim)
        
    def forward(self, x):
        
        
        # h0
        h0 = Variable(torch.zeros(2*self.layer_num, x.size(0), self.hidden_dim)).to(device)
        
        # c0
        c0 = Variable(torch.zeros(2*self.layer_num, x.size(0), self.hidden_dim)).to(device)
        
        # embedding
        x = self.embedding(x)
        # lstm
        x, (hn ,cn) = self.lstm(x, (h0, c0))
        x = self.dropout(x)
        
        # output layer
        x = self.fc(x[:, -1, :])
        
        return x
        
        
def save_best(is_best, best_accuracy, model, epoch, path):
    filename = path + 'best_model.pth'
    
    if is_best:
        if not osp.exists(path):
            os.makedirs(path)
        torch.save({'epoch':epoch,
                   'model_state_dict':model.state_dict(),
                    'best_accuracy':best_accuracy
                   }, filename)
        
        print(best_accuracy)
