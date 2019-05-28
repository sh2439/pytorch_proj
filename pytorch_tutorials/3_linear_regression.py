'''
Step 1: Create model class
'''
class LinearRegressionModel(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
        
    def forward(self, x):
        output = self.linear(x)
        return output
    
"""
Step 2: Instantiate model class
"""
input_dim = 1
output_dim =1
model = LinearRegressionModel(input_dim, output_dim)
    
"""
Step 3: Instantiate loss
"""
criterion = nn.MSELoss()

"""
Step 4: Instantiate the optimizer
"""
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(),lr = learning_rate)

"""
Step 5: Train the model
"""
epochs = 100
for epoch in range(epochs):
    epoch +=1
    
    # numpy to tensor
    inputs = Variable(torch.from_numpy(x_train))
    labels = Variable(torch.from_numpy(y_train))
    
    # clear the gradients
    optimizer.zero_grad()
    
    # forward
    outputs = model(inputs)
    
    # compute the loss
    loss = criterion(outputs, labels)
    
    # Getting gradients
    loss.backward()
    
    # update
    optimizer.step()
    
    print('epoch {}, loss {}'.format(epoch,loss.item()))