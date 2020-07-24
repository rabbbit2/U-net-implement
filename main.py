#%%
import torch
import torchvision
import PIL
from IPython.display import display
import numpy
import matplotlib.pyplot as plt
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

c=segFolder("C:\\Users\\Sharron\\Desktop\\pytorch\\ttt\\Unet-master\\train",both_transform=Compose([
    [torchvision.transforms.Grayscale(),torchvision.transforms.Grayscale()],
    [torchvision.transforms.Resize(500),torchvision.transforms.Resize(500)],
    RandomCrop(284,padding=(92,92,92,92),padding_mode="symmetric",pad_if_needed=False),
    [None,torchvision.transforms.CenterCrop(100)],
    [torchvision.transforms.ToTensor(),torchvision.transforms.ToTensor()],
    [None,toint64],
    [None,torch.squeeze]
]))

#%%

a=segFolder("C:\\Users\\Sharron\\Desktop\\pytorch\\Unet-master\\train",both_transform=Compose([
    [torchvision.transforms.Grayscale(),torchvision.transforms.Grayscale()],
    [torchvision.transforms.Resize(500),torchvision.transforms.Resize(500)],
    [torchvision.transforms.ToTensor(),None]

]))



def W(label,w0):
    label=torch.Tensor.to(label,dtype=torch.float32)
    #a=torch.sum(label)
    #b=n-a
    #labelc=n*(label*((1/a)-(1/b))+(1/b))/2
    label=10*label+w0*numpy.exp(-(ndimage.distance_transform_edt(label)**2)/50)
    return torch.Tensor.to(label,dtype=torch.float32)


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

for i in range(1500):
    
    for index ,j in enumerate(torch.utils.data.DataLoader(c,batch_size=14)):
        y_pred=Model(j[0].cuda())
        loss = torch.mean(criterion(y_pred, j[1].cuda()))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(i,"loss=",loss.item())

# %%
Model.eval().cpu()
with torch.no_grad():
    b=evalmodel(a[0][0],Model,284,100)

torchvision.transforms.ToPILImage()(1-b).save("C:\\Users\\Sharron\\Desktop\\pytorch\\present\\nwlossresult.png")
a[0][1].save("C:\\Users\\Sharron\\Desktop\\pytorch\\present\\true.png")

#%%
defor=segFolder("C:\\Users\\Sharron\\Desktop\\pytorch\\Unet-master\\train",both_transform=Compose([
    [torchvision.transforms.Grayscale(),torchvision.transforms.Grayscale()],
    [torchvision.transforms.Resize(500),torchvision.transforms.Resize(500)],
    [numpy.array,numpy.array],
    lambda X: deform_random_grid(X,10,5),
    [torchvision.transforms.ToPILImage(),torchvision.transforms.ToPILImage()]
]))
dd=defor.__getitem__(0)
#%%
dd[1].save("C:\\Users\\Sharron\\Desktop\\pytorch\\present\\defolab.png")
dd[0].save("C:\\Users\\Sharron\\Desktop\\pytorch\\present\\defo.png")
# %%
torch.cuda.empty_cache()
Model.train().cuda()
#%%
evalmodel
W(j[1],0.5).cuda()*