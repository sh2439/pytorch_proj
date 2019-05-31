### The following code implements the RNN on MNIST dataset in pytorch.

import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as dsets
import torchvision.transforms as transforms

### Load the data
train_dataset = dsets.MNIST(root = './data',
                           download = True,
                           train = True,
                           transform = transforms.ToTensor()
                           )

test_dataset = dsets.MNIST(root = './data',
                           download = True,
                           train = False,
                           transform = transforms.ToTensor()
                           )

### Make dataset iterable
iters = 3000
batch_size = 100
num_epochs = (int)(iters/(len(train_dataset)/batch_size))

train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                          batch_size = batch_size,
                                          shuffle = True)
test_loader = torch.utils.data.DataLoader(dataset= test_dataset,
                                         batch_size = batch_size,
                                         shuffle = False)

### Build the model class
### 2 hidden layers
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class RNNA(nn.Module):
    def __init__(self,input_dim, hidden_dim, output_dim, layer_dim):
        super(RNNA,self).__init__()
        
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        # Build the RNN
        self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, nonlinearity = 'tanh', batch_first = True)
        # readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        
        # Initialize the hidden layer
        h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)).to(device)
        
        out,hn = self.rnn(x, h0)
        
        # Index the hidden state of last time step
        out = self.fc(out[:,-1,:])
        
        return out

        
### Instantiate the model class
input_dim = 28
hidden_dim = 100
layer_dim = 2
output_dim = 10

model = RNNA(input_dim = 28, hidden_dim = 100, output_dim = 10, layer_dim = 2)
model.to(device)
        
### Instantiate the loss
criterion = nn.CrossEntropyLoss()

### Innstantiate the optimizer
learning_rate = 0.1
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)


### Train the model
seq_len = 28
iter_count = 0

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        #load the data
        images = Variable(images.view(-1,seq_len, input_dim)).to(device)
        labels = Variable(labels).to(device)
        
        # clear grads buffer
        optimizer.zero_grad()
        
        # compute the output
        outputs = model(images)
        
        # loss
        loss = criterion(outputs, labels)
        
        # grad
        loss.backward()
        
        # update the para
        optimizer.step()
        
        iter_count+=1
    
    ## Calculate the test acc every epoch
    total = 0
    correct = 0
    for images, labels in test_loader:
        images = Variable(images.view(-1, seq_len, input_dim)).to(device)
        
        outputs = model(images)
        
        _, preds = torch.max(outputs.data, 1)
        
        correct += (preds.to('cpu') == labels).sum()
        total += labels.size(0)
        
    accuracy = 100* correct.item()/total
    print('epoch {}, loss {}, accuracy {}'.format(epoch+1, loss.item(), accuracy))
        
    