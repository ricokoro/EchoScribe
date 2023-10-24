import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms

#format the data
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# "one hot" formatting
target_transform = transforms.Compose(
    [transforms.Lambda(lambda x: F.one_hot(torch.tensor(x), 10))])

batch_size = 32

# (need to replace with out training set) trainset = torchvision.datasets.CIFAR10(root='data', train=True, download=True, transform=transform, target_transform=target_transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size)

# (need to replace with out test set) testset = torchvision.datasets.CIFAR10(root='data', train=False, download=True, transform=transform, target_transform=target_transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size)

# The classes of our data
classes = ('good', 'hello', 'man', 'girl', 'sorry')

# everything below is from CNN example code

'''

class CNNModel(nn.Module):
    def __init__(self):
        # call the parent constructor
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 20, kernel_size=(5, 5)) #number of channels = number of inputs
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=128, kernel_size=(5, 5))
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(5, 5))
        self.conv4 = nn.Conv2d(in_channels=64, out_channels = 100, kernel_size=(5, 5))
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.fc1 = nn.Linear(in_features=100, out_features=50)
        self.dropout2 = nn.Dropout(0.1) #where we fix some of the weights and don't optimize them
        self.fc2 = nn.Linear(in_features=50, out_features=10)
    def forward(self, x):
        # pass the input through our first set of CONV => RELU =>
        # POOL layers
        x = self.conv1(x)
        x = F.relu(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.maxpool2(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        output = self.fc2(x)
        # return the output predictions
        return output

model = CNNModel()

# define training hyperparameters
lr = 1e-2
num_epochs = 20

# set the device we will be using to train the model (to enable hardware acceleration)
# uncomment if you have a cuda supported gpu
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# or uncomment if on a m1/m2 mac
#device = torch.device("mps" if torch.backends.mps.is_available() else "cpu") 

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr)

model.to(device)

for e in range(0, num_epochs):
	# set the model in training mode
	model.train()
	# initialize the total training and validation loss
	totalTrainLoss = 0
	totalValLoss = 0
	# initialize the number of correct predictions in the training
	# and validation step
	trainCorrect = 0
	valCorrect = 0
	# loop over the training set
	for i, (x, y) in enumerate(trainloader):
		# send the input to the device
		(x, y) = (x.to(device), y.to(torch.float32).to(device))
		# perform a forward pass and calculate the training loss
		pred = model(x)
		loss = criterion(pred, y)
		# zero out the gradients, perform the backpropagation step,
		# and update the weights
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		# add the loss to the total training loss so far and
		# calculate the number of correct predictions
		totalTrainLoss += loss
	print("Epoch", e, "Training Loss:", totalTrainLoss.item())
      
#track test loss
import numpy as np
test_loss = 0.0
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

model.eval()
# iterate over test data
for i, (x, y) in enumerate(testloader):
	# send the input to the device
	(x, y) = (x.to(device), y.to(torch.float32).to(device))
	output = model(x)
	# calculate the batch loss
	loss = criterion(output, y)
	# update test loss 
	test_loss += loss.item() * x.size(0)
	# convert output probabilities to predicted class
	_, pred = torch.max(output, 1)   
	_, y = torch.max(y, 1)    
	# compare predictions to true label
	correct_tensor = pred.eq(y)
	correct = np.squeeze(correct_tensor)
	# calculate test accuracy for each object class
	for i in range(len(y.data)):
		label = y.data[i]
		class_correct[label] += correct[i].item() 
		class_total[label] += 1

# average test loss
test_loss = test_loss/len(testloader.dataset)
print('Test Loss: {:.6f}\n'.format(test_loss))

for i in range(10):
    if class_total[i] > 0:
        print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
            classes[i], 100.0 * class_correct[i] / class_total[i],
            np.sum(class_correct[i]), np.sum(class_total[i])))
    else:
        print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
    100. * np.sum(class_correct) / np.sum(class_total),
    np.sum(class_correct), np.sum(class_total)))

'''