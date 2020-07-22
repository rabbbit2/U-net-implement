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
import .custom as *

#%%
c=segFolder("ttt\\Unet-master\\train",both_transform=Compose([
    [torchvision.transforms.Grayscale(),torchvision.transforms.Grayscale()],
    RandomCrop(188,padding=(92,92,92,92),padding_mode="symmetric",pad_if_needed=False),
    [None,torchvision.transforms.CenterCrop(4)],
    [torchvision.transforms.ToTensor(),torchvision.transforms.ToTensor()],
    [None,toint64],
    [None,torch.squeeze]
]))

#%%
#%%
criterion = torch.nn.CrossEntropyLoss()
optimizer=torch.optim.SGD(Model.parameters(),momentum=0.9,lr=0.00005)
accuracy=numpy.zeros((29))
lossinfor=numpy.zeros((29))
accuracyPerepoch0=[]
lossPerepoch0=[]

for i in range(500000):
    
    for index ,j in enumerate(torch.utils.data.DataLoader(c,batch_size=29)):
        y_pred=Model(j[0].cuda())
        loss = criterion(y_pred, j[1].cuda())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(i,"loss=",loss.item())