### The following code implements neural network on MNIST data in pytorch.
### 2 Hidden Layer Feedforward Neural Network (ReLU Activation)
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms


"""
Step 1: loading the dataset
"""

train_dataset = dsets.MNIST(root = './data',
                           train = True,
                           transform = transforms.ToTensor(),
                           download = True)

test_dataset = dsets.MNIST(root = './data',
                           train = False,
                           transform = transforms.ToTensor(),
                           download = True)

"""
Step 2: Making dataset iterable
"""
batch_size = 100
n_iters = 6000
num_epochs = int(n_iters/(len(train_dataset)/batch_size))

train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                         batch_size = batch_size,
                                         shuffle = True)

test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                         batch_size = batch_size,
                                         shuffle = False)

"""
Step 3: Create the model class
"""
# 784 - 100 - 100 - 10
class FeedforwardModelD(nn.Module):
    def __init__(self, input_size, hidden_size,output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, output_size)
        
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x
"""
Step 4: Instantiate model class
"""

input_dim = 28*28
hidden_dim = 100
output_dim = 10
model = FeedforwardModelD(input_dim, hidden_dim, output_dim)

"""
Step 5: Instantiate loss class
"""
criterion = nn.CrossEntropyLoss()


"""
Step 6: Instantiate optimizer class
"""
learning_rate = 0.01
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    
"""
Step 7: Train the model
"""
iter_count = 0
for epoch in range(num_epochs):
    
    for i, (images, labels) in enumerate(train_loader):
        # load images as variables
        images = Variable(images.view(-1,28*28))
        labels = Variable(labels)
        
        # clear the gradients
        optimizer.zero_grad()
        
        # compute the output
        outputs = model(images)
        
        # compute the loss
        loss = criterion(outputs, labels)
        
        # compute the gradients
        loss.backward()
        
        optimizer.step()
        
        iter_count +=1
        
        if iter_count % 500 ==0:
            correct = 0
            total = 0
            
            # iterate through test
            for images, labels in test_loader:
                images = Variable(images.view(-1,28*28))
                
                outputs = model(images)
                _,preds = torch.max(outputs.data, 1)
                
                total += labels.size()[0]
                correct += (labels == preds).sum()
                
            accuracy = float(100*correct.item()/total)
            
            # print loss
            print('iteration {}, loss {}, accuracy {}'.format(iter_count, loss.item(), accuracy))
            
 