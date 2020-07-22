#%%
import torch

#%% 
def cropcon(a,b):
    size=int((a.shape[3]-b.shape[3])/2)
    return torch.cat((a[:,:,size:(b.shape[3]+size),size:(b.shape[3]+size)],b),1)
#%%
class Umodel(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1_1=torch.nn.Conv2d(1,64,3)
        self.conv1_2=torch.nn.Conv2d(64,64,3)

        self.conv2_1=torch.nn.Conv2d(64,128,3)
        self.conv2_2=torch.nn.Conv2d(128,128,3)

        self.conv3_1=torch.nn.Conv2d(128,256,3)
        self.conv3_2=torch.nn.Conv2d(256,256,3)

        self.conv4_1=torch.nn.Conv2d(256,512,3)
        self.conv4_2=torch.nn.Conv2d(512,512,3)

        self.conv5_1=torch.nn.Conv2d(512,1024,3)
        self.conv5_2=torch.nn.Conv2d(1024,1024,3)
        self.upconv1=torch.nn.ConvTranspose2d(1024,512,2,stride=2)

        self.conv6_1=torch.nn.Conv2d(1024,512,3)
        self.conv6_2=torch.nn.Conv2d(512,512,3)
        self.upconv2=torch.nn.ConvTranspose2d(512,256,2,stride=2)

        self.conv7_1=torch.nn.Conv2d(512,256,3)
        self.conv7_2=torch.nn.Conv2d(256,256,3)
        self.upconv3=torch.nn.ConvTranspose2d(256,128,2,stride=2)

        self.conv8_1=torch.nn.Conv2d(256,128,3)
        self.conv8_2=torch.nn.Conv2d(128,128,3)
        self.upconv4=torch.nn.ConvTranspose2d(128,64,2,stride=2)

        self.conv9_1=torch.nn.Conv2d(128,64,3)
        self.conv9_2=torch.nn.Conv2d(64,64,3)

        self.conv10=torch.nn.Conv2d(64,2,1)

        #self.softm=torch.nn.Softmax2d()



    def forward(self,input):

        c1=torch.nn.functional.relu((self.conv1_2(torch.nn.functional.relu(self.conv1_1(input)))))

        c2=torch.nn.functional.relu(self.conv2_2(torch.nn.functional.relu(self.conv2_1(torch.nn.MaxPool2d(2)(c1)))))

        c3=torch.nn.functional.relu(self.conv3_2(torch.nn.functional.relu(self.conv3_1(torch.nn.MaxPool2d(2)(c2)))))

        c4=torch.nn.functional.relu(self.conv4_2(torch.nn.functional.relu(self.conv4_1(torch.nn.MaxPool2d(2)(c3)))))

        e4=self.upconv1(torch.nn.functional.relu(self.conv5_2(torch.nn.functional.relu(self.conv5_1(torch.nn.MaxPool2d(2)(c4))))))

        e3=self.upconv2(torch.nn.functional.relu(self.conv6_2(torch.nn.functional.relu(self.conv6_1(cropcon(c4,e4))))))

        e2=self.upconv3(torch.nn.functional.relu(self.conv7_2(torch.nn.functional.relu(self.conv7_1(cropcon(c3,e3))))))

        e1=self.upconv4(torch.nn.functional.relu(self.conv8_2(torch.nn.functional.relu(self.conv8_1(cropcon(c2,e2))))))

        output=self.conv10(torch.nn.functional.relu(self.conv9_2(torch.nn.functional.relu(self.conv9_1(cropcon(c1,e1))))))

        #output=torch.nn.LogSoftmax(output,1)

        return output