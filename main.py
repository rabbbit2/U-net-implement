#%%
import torch
import torchvision
import PIL
from IPython.display import display
import numpy
import matplotlib.pyplot
import os
import os.path
from PIL import Image
from scipy import ndimage
import seaborn as sns
import pandas as pd
from custom import *
from elasticdeform import deform_random_grid
import model

#%%
def toint64(tensor):
    return torch.Tensor.to(tensor,dtype=torch.int64)

c=segFolder("ttt\\Unet-master\\train",both_transform=Compose([
    [torchvision.transforms.Grayscale(),torchvision.transforms.Grayscale()],
    RandomCrop(188,padding=(92,92,92,92),padding_mode="symmetric",pad_if_needed=False),
    [None,torchvision.transforms.CenterCrop(4)],
    [torchvision.transforms.ToTensor(),torchvision.transforms.ToTensor()],
    [None,toint64],
    [None,torch.squeeze]
]))

#%%
'''
c=segFolder("ttt\\Unet-master\\train",both_transform=Compose([
    [torchvision.transforms.Grayscale(),torchvision.transforms.Grayscale()],
    RandomCrop(188,padding=(92,92,92,92),padding_mode="symmetric",pad_if_needed=False),
    [lambda X: numpy.array(X),lambda X: numpy.array(X)],
    lambda X:deform_random_grid(X,sigma=10,points=3),
    [None,lambda X: (X>250)*1]
]))

'''
a=c.__getitem__(0)

#def W(label,w0,sigma):


#%%
def weight_init(m):
    if isinstance(m,model.Umodel)==False:
        std=numpy.sqrt(2/numpy.prod(m.weight.shape[1:4]))
        torch.nn.init.normal_(m.weight,std=std)
Model=model.Umodel().cuda()
model.Umodel.apply(Model,weight_init)
#%%
criterion = torch.nn.CrossEntropyLoss(reduction="none")
optimizer=torch.optim.SGD(Model.parameters(),momentum=0.99,lr=0.005)
accuracy=numpy.zeros((29))
lossinfor=numpy.zeros((29))
accuracyPerepoch0=[]
lossPerepoch0=[]

for i in range(500000):
    
    for index ,j in enumerate(torch.utils.data.DataLoader(c,batch_size=29)):
        y_pred=Model(j[0].cuda())
        loss = torch.mean(criterion(y_pred, j[1].cuda()))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(i,"loss=",loss.item())

# %%


evalmodel

# %%
