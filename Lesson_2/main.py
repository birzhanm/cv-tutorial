"""
This file introduces CNN architecture and attempts to answer the following questions
1. What is a CNN?
2. How can I build a simple CNN using Numpy library?
3. How can I build a simple CNN using one of the Deep Learning frameworks, say PyTorch :)?
"""

"""
Question 1: What is a CNN?
Roughly speaking a convolutional neural network (CNN) is a neural network using a convolutional layers
for features extraction. What is a convolutional layer, one should ask? I guess, I cannot explain this
concept better than Adit Deshpande, who has written a series of amazing posts on CNNs. Make sure to check out
his posts on the following link:
https://adeshpande3.github.io/A-Beginner%27s-Guide-To-Understanding-Convolutional-Neural-Networks/
The working link is included in Resources.md file for this repo.
"""

"""
Question 2: How can one build a simple CNN using Numpy?

Let's try model the effect of a convolutional filter on an image.
Here are our inputs:

image: a Numpy array representing a given image. Shape of image array is
(image_size)x(image_size)x(3), i.e. we assume a colored square image.
filter: a square filter of size (filter_size)x(filter_size)x(3). We assume that filter_size is odd, say 3,5 or 7.

Here are functions we wish to build:

im2col: function unrolling a filter volume of size (filter_size)x(filter_size)x(3) into a column of length
filter_size*filter_size*3 so that convolution operation could be reduced to the sum of dot product
and of bias term
filter2col: unrolls a three dimensional filter into a column vector
conv: computing the actual convolution operation
"""
image_size = 10
filter_size = 3

import numpy as np
def im2col(image, filter_size):
    image_size = image.shape[0]
    pad = (filter_size - 1)//2
    output = np.zeros((image_size, image_size, filter_size*filter_size*3), dtype=float)
    for i in range(pad, image_size-pad):
        for j in range(pad, image_size-pad):
            output[i,j,:] = image[i-pad:i+pad+1,j-pad:j+pad+1,:].reshape(filter_size*filter_size*3, order='F')
            # order 'F' means that the last coordinate changes the slowest during reshaping
    return output

def filter2col(filter):
    filter_size = filter.shape[0]
    return filter.reshape(filter_size*filter_size*3, order='F')


def conv(image,filter,bias):
    filter_size = filter.shape[0]
    im_col = im2col(image, filter_size)
    filter_col = filter2col(filter)
    filter_result = np.dot(im_col, filter_col)
    conv_result = filter_result + bias
    return conv_result



# sanity check for the above functions
image_size = 5
filter_size = 3
image = np.random.rand(image_size, image_size, 3)
filter = np.random.rand(filter_size, filter_size, 3)
bias = np.random.rand(1)
print(im2col(image, filter_size).shape)
print(filter2col(filter).shape)
print(conv(image, filter, bias).shape)
"""
Remark:
We have done a forward propagattion for a single convolutional layer. It required a bit of work, and writing
backpropagation also requires some (a bit more) work. Fortunately, deep learning libraries make forward propagation
easier and eliminate the need for backpropagation altogether. Let's try to build some simple CNNs using PyTorch,
one of the popular Deep Learning frameworks
"""

"""
How can I build a simple CNN using one of the Deep Learning frameworks, say PyTorch :)?
"""

import torch
import torchvision
import torchvision.transforms as transforms

"""
Load MNIST dataset.
This dataset contains grayscale images of size 28x28.
"""

# download and prepare data
# set a manual seed for reproducibility of results.
torch.manual_seed(7)
# set a batch size for training models
batch_size = 8

# define transform on a dataset
# the values for mean and standard deviation for MNIST dataset is from 'https://nextjournal.com/gkoehler/pytorch-mnist'.
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,),(0.3081,))])
# load MNIST data available in torchvision
train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
# prepare batches for both train_set and test_set
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

# let us look at some of the training images
import matplotlib.pyplot as plt
import numpy as np

#roll batches
batches = enumerate(train_loader)
batch_id, (images, labels) = next(batches)

print(images)

fig = plt.figure()
for i in range(8):
  plt.subplot(3,4,i+1)
  plt.tight_layout()
  plt.imshow(images[i][0], cmap='gray', interpolation='none')
  plt.title("Ground truth: {}".format(labels[i]))
  plt.xticks([])
  plt.yticks([])

"""
Design a simple CNN
"""
# simple CNN
import torch.nn as nn
import torch.nn.functional as F

class CNN_V1(nn.Module):
  def __init__(self):
    super(CNN_V1,self).__init__()
    self.conv = nn.Conv2d(1,3,3)
    self.fc = nn.Linear(3*26*26,10)

  def forward(self, x):
    x = F.relu(self.conv(x))
    x = x.view(-1, 3*26*26)
    x = self.fc(x)
    return x

"""
Construct a bit more complex CNN
"""
# adding more layers
import torch.nn as nn
import torch.nn.functional as F

class CNN_V2(nn.Module):
  def __init__(self):
    super(CNN_V2,self).__init__()
    self.conv1 = nn.Conv2d(1,3,3)
    self.conv2 = nn.Conv2d(3,6,3)
    self.fc1 = nn.Linear(6*24*24,100)
    self.fc2 = nn.Linear(100,10)


  def forward(self, x):
    x = F.relu(self.conv1(x))
    x = F.relu(self.conv2(x))
    x = x.view(-1, 6*24*24)
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return x

"""
Initiate both models
"""
# set up number of epochs
epochs = 2

# initiate models
cnn_v1 = CNN_V1()
cnn_v2 = CNN_V2()

# let's do a single forward prop as a sanity check
x = torch.rand(2,1,28,28)
out_v1 = cnn_v1(x)
out_v2 = cnn_v2(x)

print("Output for V1: {}".format(out_v1))
print("Output for V2: {}".format(out_v2))


"""
Train cnn_v1
"""
# define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn_v1.parameters(), lr=0.001)

# prepare GPU for training
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# wrap the model into gpu if there is one
cnn_v1.to(device)

# training loop
for epoch in range(epochs):
  running_loss = 0.0
  for i, data in enumerate(train_loader, 0):
    # get inputs
    inputs, labels = data
    inputs = inputs.to(device)
    labels = labels.to(device)

    # zero parameter gradients
    optimizer.zero_grad()

    # forward + backward + optimize
    outputs = cnn_v1(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    # printing statistics
    running_loss += loss.item()
    if i%200 == 199:
      print('Epoch: {}, Batch: {} - Loss: {}'.format(epoch, i+1, running_loss/200))
      running_loss = 0
print('Finished Training')

"""
Train cnn_v2
"""
# define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn_v2.parameters(), lr=0.001)

# wrap the model into gpu
cnn_v2.to(device)

# training loop
for epoch in range(epochs):
  running_loss = 0.0
  for i, data in enumerate(train_loader, 0):
    # get inputs
    inputs, labels = data
    inputs = inputs.to(device)
    labels = labels.to(device)

    # zero parameter gradients
    optimizer.zero_grad()

    # forward + backward + optimize
    outputs = cnn_v2(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    # printing statistics
    running_loss += loss.item()
    if i%200 == 199:
      print('Epoch: {}, Batch: {} - Loss: {}'.format(epoch, i+1, running_loss/200))
      running_loss = 0
print('Finished Training')

"""
Test cnn_v1
"""
total = 0
correct = 0
with torch.no_grad():
  for data in test_loader:
    inputs, labels = data
    inputs = inputs.to(device)
    labels = labels.to(device)
    outputs = cnn_v1(inputs)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()
print('Accuracy of cnn_v1 on 10,000 test images is {}'.format(correct/total))

"""
Test cnn_v2
"""
total = 0
correct = 0
with torch.no_grad():
  for data in test_loader:
    inputs, labels = data
    inputs = inputs.to(device)
    labels = labels.to(device)
    outputs = cnn_v2(inputs)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()
print('Accuracy of cnn_v2 on 10,000 test images is {}'.format(correct/total))
