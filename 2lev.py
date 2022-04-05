from __future__ import division, print_function
import torchvision.datasets as dsets
import torch
import torch.nn as nn
from torchvision.utils import save_image
from torchvision import models, transforms
import numpy as np
import matplotlib.pyplot as plt

# original data
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

Train = dsets.ImageFolder('...', transform) #put your training data path
Test = dsets.ImageFolder('...', transform) #put your testing data path


train_loader = torch.utils.data.DataLoader(dataset=Train, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=Test, batch_size=64, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1, stride=1) 
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2, return_indices=True) 

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=1) 
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2, return_indices=True) 

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1) 
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, 2, return_indices=True)  

        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1) 
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(2, 2, return_indices=True) 

        self.conv5 = nn.Conv2d(128, 256,  kernel_size=3, padding=1, stride=1) 
        self.relu5 = nn.ReLU()
        self.pool5 = nn.MaxPool2d(2, 2, return_indices=True)

        self.conv6 = nn.Conv2d(256, 512, kernel_size=7, stride=1) #BottleNeck
        self.relu6 = nn.ReLU()

        # deconv

        self.Convt1 = nn.ConvTranspose2d(512, 256, kernel_size=7, stride=1) 
        self.relu_Convt1 = nn.ReLU()
        self.unpool1 = nn.MaxUnpool2d(kernel_size=2, stride=2) 

        self.conv7 = nn.Conv2d(512, 256, kernel_size=3, padding=1, stride=1)  
        self.relu7 = nn.ReLU()
        self.conv8 = nn.Conv2d(256, 128, kernel_size=3, padding=1, stride=1) 
        self.relu8 = nn.ReLU()
        self.unpool2 = nn.MaxUnpool2d(kernel_size=2, stride=2) 

        self.conv9 = nn.Conv2d(256, 128, kernel_size=3, padding=1, stride=1)  
        self.relu9 = nn.ReLU()
        self.conv10 = nn.Conv2d(128, 64, kernel_size=3, padding=1, stride=1)
        self.relu10 = nn.ReLU()
        self.unpool3 = nn.MaxUnpool2d(kernel_size=2, stride=2)  

        self.conv11 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1) 
        self.relu11 = nn.ReLU()
        self.conv12 = nn.Conv2d(64, 32, kernel_size=3, padding=1, stride=1)
        self.relu12 = nn.ReLU()
        self.unpool4 = nn.MaxUnpool2d(kernel_size=2, stride=2)  # 112

        self.conv13 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)  
        self.relu13 = nn.ReLU()
        self.conv14 = nn.Conv2d(32, 16, kernel_size=3, padding=1, stride=1)
        self.relu14 = nn.ReLU()
        self.unpool5 = nn.MaxUnpool2d(kernel_size=2, stride=2)

        self.conv15 = nn.Conv2d(32, 16, kernel_size=3, padding=1, stride=1)
        self.relu15 = nn.ReLU()
        self.conv16 = nn.Conv2d(16, 1, kernel_size=3, padding=1, stride=1)

    def forward(self, x):
        # encoder
        con1 = self.conv1(x)
        x = self.relu1(con1)
        x, indices1 = self.pool1(x)

        con2 = self.conv2(x)
        x = self.relu2(con2)
        x, indices2 = self.pool2(x)

        con3 = self.conv3(x)
        x = self.relu3(con3)
        x, indices3 = self.pool3(x)

        con4 = self.conv4(x)
        x = self.relu4(con4)
        x, indices4 = self.pool4(x)

        con5 = self.conv5(x)
        x = self.relu5(con5)
        x, indices5 = self.pool5(x)

        x = self.conv6(x) #Final Compression
        latent = x #Latent Features
        x = self.relu6(x)

        # decoder
        x = self.Convt1(x)
        x = self.relu_Convt1(x)
        x = self.unpool1(x, indices=indices5)

        concat_1 = torch.cat([x, con5], dim=1)
        x = self.conv7(concat_1)
        x = self.relu7(x)
        x = self.conv8(x)
        x = self.relu8(x)
        x = self.unpool2(x, indices=indices4)

        concat_2 = torch.cat([x, con4], dim=1)
        x = self.conv9(concat_2)
        x = self.relu9(x)
        x = self.conv10(x)
        x = self.relu10(x)
        x = self.unpool3(x, indices=indices3)

        concat_3 = torch.cat([x, con3], dim=1)
        x = self.conv11(concat_3)
        x = self.relu11(x)
        x = self.conv12(x)
        x = self.relu12(x)
        x = self.unpool4(x, indices=indices2)

        concat_4 = torch.cat([x, con2], dim=1)
        x = self.conv13(concat_4)
        x = self.relu13(x)
        x = self.conv14(x)
        x = self.relu14(x)
        x = self.unpool5(x, indices=indices1)

        concat_5 = torch.cat([x, con1], dim=1)
        x = self.conv15(concat_5)
        x = self.relu15(x)
        x = self.conv16(x)
        return x, latent


      
### model: THe network that will be newly trained
### load = Pretrained model

model = ConvAutoencoder() 
model.to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

load = # load pretrained model
load.to(device)
training_epochs = 20
ToT_train_loss = []
print()
print('2lev CAE Start')
print()
for epoch in range(1, training_epochs + 1):
    model.train()
    train_loss = 0.0
    i=0
    for data in train_loader:

        train_image, train_label = data
        train_image = train_image.to(device)

        output_rec, features = load(train_image)
        output_rec = output_rec.to(device)

        optimizer.zero_grad()
        
        train_output, _ = model(output_rec)
        loss= criterion(train_output, output_rec) ##***TRAIN LOSS PART****

        loss.backward()
        optimizer.step()
        train_loss += loss.item() * train_image.size(0)

    # print avg training statistics
    train_loss = train_loss / len(train_loader)
    ToT_train_loss.append(train_loss)
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))


save = #SAVE SECOND MODEL

