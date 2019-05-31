### The following code implements the LSTM Network on MNIST dataset in pytorch.


import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from tqdm import tqdm

"""
Step 1: load the dataset
"""
train_dataset = dsets.MNIST(root = './data',
                           train = True,
                            transform = transforms.ToTensor(),
                           download = True
                           )
test_dataset = dsets.MNIST(root = './data',
                          train = False,
                          transform = transforms.ToTensor(),
                          download = True)

"""
Step 2: make the dataset iterable
"""
batch_size = 100
n_iters = 6000
num_epochs = (int)(n_iters/(len(train_dataset)/batch_size))

train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                          batch_size = batch_size,
                                          shuffle = True)

test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                         batch_size = batch_size,
                                         shuffle = False)

"""
Step 3: Build the model class
"""

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
class LSTM_ModelA(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTM_ModelA, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        
        # lstm layer
        # (batch, seq, feature)
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first = True)
        
        # readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self,x):
        
        # initialize the hidden state with zeros
        h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)).to(device)
    
        # initialize the cell state
        c0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)).to(device)
        
        # one time step
        output, (hn, cn) = self.lstm(x, (h0, c0))
        
        # only use the last time step
        out = self.fc(output[:,-1,:])
        
        return out

"""
Step 4: Instantiate the model class 
"""
input_dim = 28
hidden_dim = 100
seq_dim = 28
layer_dim = 3
output_dim = 10

modelA = LSTM_ModelA(input_dim, hidden_dim, layer_dim, output_dim)

modelA.to(device)
"""
Step 5: Instantiate the loss
"""
criterion = nn.CrossEntropyLoss()

"""
Step 6: Instantiate the optimizer
"""
learning_rate = 0.1
optimizer = torch.optim.SGD(modelA.parameters(), lr = learning_rate)

# check model parameters
for para in modelA.state_dict():
    print(para, modelA.state_dict()[para].size())
    
    
"""
Step 7: Train the model
"""
iter_count = 0
for epoch in tqdm(range(num_epochs)):
    epoch += 1
    
    for i, (images, labels) in enumerate(train_loader):
        # load data
        images = Variable(images.view(-1,seq_dim, input_dim)).to(device)
        labels = Variable(labels).to(device)
        
        # clear grad buffers
        optimizer.zero_grad()
        
        # compute outputs
        outputs = modelA(images)
        
        # compute loss
        loss = criterion(outputs, labels)
        # backward
        loss.backward()
        # update
        optimizer.step()
        
        iter_count+=1
        
        if iter_count%500 ==0:
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = Variable(images.view(-1, seq_dim, input_dim)).to(device)
                outputs = modelA(images)
                _,preds = torch.max(outputs.data, dim = 1)
                
                total += labels.size(0)
                correct += float((preds.to('cpu') == labels).sum())
                
            accuracy = 100* correct/total
            
            print('epoch {}, iters {}, loss {}, acc {}'.format(epoch,iter_count, loss.item(), accuracy))
                
        