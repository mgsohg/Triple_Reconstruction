from __future__ import division, print_function
from Backbone import *
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

      
### model: 3levCAE
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
        loss = criterion(train_output, train_image) ##***TRAIN LOSS PART****

        loss.backward()
        optimizer.step()
        train_loss += loss.item() * train_image.size(0)

    # print avg training statistics
    train_loss = train_loss / len(train_loader)
    ToT_train_loss.append(train_loss)
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))


save = #SAVE Third MODEL
