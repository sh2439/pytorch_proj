### The following code implements the linear regression on MNIST dataset in pytorch.

"""
Step 1: Load the data
"""
train_dataset = dsets.MNIST(root = './data',
                          train = True,
                           transform = transforms.ToTensor(),
                           download = True

                          )

test_dataset = dsets.MNIST(root = './data',
                          train = False,
                           transform = transforms.ToTensor(),
                           download = True

                          )

"""
Step 2: Make the dataset iterable
"""

batch_size = 100
n_iters = 6000
num_epochs = n_iters/(len(train_dataset)/batch_size)

train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                    batch_size = batch_size,
                                    shuffle = True)

test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                   batch_size = batch_size,
                                   shuffle = False)

"""
Step 3: Build the model class
"""
class CnnModelD(nn.Module):
    def __init__(self):
        super().__init__()
        
        # conv1 - avgpool - relu
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 16, kernel_size = 5, padding = 2, stride = 1)
        self.maxpool1 = nn.MaxPool2d(kernel_size = 2)
        self.relu1 = nn.ReLU()
        
        self.drop1 = nn.Dropout(0.8)
        
        # conv2 - avgpool - relu
        self.conv2 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 5, padding = 2, stride = 1)
        self.maxpool2 = nn.MaxPool2d(kernel_size = 2)
        self.relu2 = nn.ReLU()
        
        # fc1
        self.fc1 = nn.Linear(7*7*32, 100)
        self.fc2 = nn.Linear(100,10)
        
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.relu1(x)
        
        x = x.drop1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.relu2(x)
        
        x = x.view(x.size(0),-1)
        x = self.fc1(x)
        x = self.fc2(x)
        
        return x

"""
Step 4: instantiate the model class
"""

model = CnnModelC()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
"""
Step 5: loss
"""
criterion = nn.CrossEntropyLoss()

"""
Step 6: optimizer
"""
learning_rate = 0.01
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 2, gamma = 0.5)
"""
Step 7: train
"""
iter_count = 0

for epoch in tqdm(range(int(num_epochs))):
    epoch  += 1
    for i, (images, labels) in enumerate(train_loader):
        
        # load images as variables
        images = Variable(images.view(-1,1,28,28)).to(device)
        
        labels = Variable(labels).to(device)
        
        # clear grads
        optimizer.zero_grad()
        
        # compute the output
        outputs = model(images)
        
        # compute the loss
        loss = criterion(outputs, labels)
        
        # compute the grads
        loss.backward()
        
        # update
        optimizer.step()
        
        iter_count +=1
        
        if iter_count % 500 ==0:
            correct = 0
            total = 0
            
            for images, labels in test_loader:
                
                images = Variable(images.view(-1,1,28,28)).to(device)
                outputs = model(images)
                
                _,preds = torch.max(outputs, 1)
                
                correct += (preds.cpu() == labels).sum()
                total += labels.size(0)
                
            accuracy = 100*correct.item()/total
            
            print( 'Epoch {}, iter {}, loss {}, accuracy {}'.format(epoch, iter_count, loss.item(), accuracy))
        




