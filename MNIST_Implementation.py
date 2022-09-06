#!/usr/bin/env python
# coding: utf-8

# In[10]:


import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim


# In[11]:


transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,)),])
# transforms.ToTensor() - converts the image into numbers.
# transforms.Normalize() - normalizes a tensor with a mean and standard deviation.


# In[12]:


trainset = datasets.MNIST(root='./data',download=True,train=False,transform=transform)

valset = datasets.MNIST(root='./data', download=True, train=False, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)

#batchsize - no. of images we want to read at a time.


# In[13]:


detaiter = iter(trainloader)
images,labels=detaiter.next()
print(images.shape)
print(labels.shape)
#64-number of images in each batch ; 28, 28 - 28*28 pixels image
#labels : 64 images with 64 labels.


# In[20]:


plt.imshow(images[1].numpy().squeeze(),cmap='copper')


# In[24]:


figure=plt.figure()
num_imgs=60
for i in range(1, num_imgs+1):
    plt.subplot(6,10,i) #6-rows; 10-columns
    plt.axis('off') #axis-switched off
    plt.imshow(images[i].numpy().squeeze(),cmap='copper') # to show the images in the given range.


# In[26]:


#Building the Neural Network: (using torch.nn)
input_size=784 #28*28
hidden_sizes=[128,64]
output_size=10

model=nn.Sequential(nn.Linear(input_size,hidden_sizes[0]),nn.ReLU(),nn.Linear(hidden_sizes[0],hidden_sizes[1]),nn.ReLU(),nn.Linear(hidden_sizes[1],output_size),nn.LogSoftmax(dim=1))
#nn.Sequential - wraps the layers in the network.
#ReLU-activation function defined as the positive part of the argument.(negative values are modified to 0)
#output lat=yer is a linear layer with LogSoftmax activation since its a classification problem
#Softmax is a mathematical functionthat converts the vector of numbers to vector of probabilities.
print(model)


# In[29]:


criterion=nn.NLLLoss() # negative log-likelihood loss
images,labels=next(iter(trainloader))
images=images.view(images.shape[0],-1)

logps=model(images) #log probabilities
loss=criterion(logps,labels) # calculate the NLL loss
print(loss)


# In[30]:


#Adjusting the weights to minimize the loss:
print("Before Backward propogation : \n",model[0].weight.grad) #default value:none
loss.backward() #to update the weights
print("After Backward propogation : \n",model[0].weight.grad)


# In[32]:


#core training process :
#We use torch.optim which optimizes the model,performs gradient-decent and updates the weights by back-propagation
#Hence each iteration reduces the training loss

optimizer=optim.SGD(model.parameters(), lr=0.003,momentum=0.9)
time0=time()
itr=15
for i in range(itr):
    run_loss=0
    for images, labels in trainloader:
        images=images.view(images.shape[0],-1) #flatten MNIST images into a 784 long vector
        
        optimizer.zero_grad() #Training pass
        output=model(images)
        loss=criterion(output,labels)
        loss.backward() #backpropagation
        optimizer.step() #optimizes the weights
        run_loss+=loss.item()
    else:
        print("Epoch {} - Training loss: {}".format(i,run_loss/len(trainloader)))
print("\nTraining Time (in min) =",(time()-time0)/60)

# As you can see in the output as the number of iteration increases the loss keeps decreasing


# In[ ]:





# In[58]:


#Testing:
images,labels=next(iter(valloader))
img=images[0].view(1,784)

with torch.no_grad():
    logps=model(img)
ps=torch.exp(logps)
prob=list(ps.numpy()[0])
plt.imshow(images[0].numpy().squeeze(),cmap='binary')
print("Predicted Digit=",prob.index(max(prob)))


# In[59]:


#Model Evaluation :
correct_count,all_count=0,0
for images,labels in valloader:
    for i in range(len(labels)):
        img=images[i].view(1,784)
        with torch.no_grad():
            logps=model(img)
        ps=torch.exp(logps)
        prob=list(ps.numpy()[0])
        pred_label=prob.index(max(prob))
        true_label=labels.numpy()[i]
        if(true_label==pred_label):
            correct_count+=1
        all_count+=1
print("Number of Images Tested =",all_count)
print("\nModel Accuracy = ",(correct_count/all_count))


# In[ ]:





# In[ ]:




