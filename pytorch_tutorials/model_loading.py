### Save and load the entire model

# Save the model
torch.save(model, PATH)

# Load the model
model = torch.load(PATH)
model.eval()



### Saving and loading a checkpoint

# Save the model/checkpoint
torch.save({'epoch':epoch,
			'model_state_dict':model.state_dict(),
			'optimizer_state_dict':optimizer.state_dict(),
			'loss':loss},
			PATH)

# Load and use the checkpoint
checkpoint = torch.load(PATH)

model = ModelClass()
optimizer = OptimizerClass()

model.load_state_dict.(checkpoint[model_state_dict])
optimizer.load_state_dict(checkpoint[optimizer_state_dict])
epoch = checkpoint['epoch']
loss = checkpoint['loss']



### Save and load model for Transfer Learning/Finetuning
torch.save(modelA.state_dict(),PATH)

modelB = ModelClass()
modelB.load_state_dict(torch.load(PATH), strict = False)


### Saving and loading models across devices

### Save on GPU, load on CPU
torch.save(model.state_dict(),PATH)

device = torch.device('cpu')
model = ModelClass()
model.load_state_dict(torch.load(PATH, map_location = device))

### Save on GPU, load on GPU
torch.save(model.state_dict(),PATH)

device = torch.device('cuda')
model = ModelClass()
model.load_state_dict(torch.load(PATH))
model.to(device)

### Save on CPU, load onn GPU
torch.save(model.state_dict(), PATH)

device = torch.device('cuda')
model = ModelClass()
model.load_state_dict(torch.load(PATH, map_location = 'cuda:0'))
model.to(device)













