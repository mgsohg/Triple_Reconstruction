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
    #transforms.Normalize(mean=[0.5], std=[0.5]),
])

Loader = dsets.ImageFolder('...', transform)
Ext_Loader = torch.utils.data.DataLoader(dataset=Loader, batch_size=1, shuffle=False)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



######Feature EXTRACTION of the first network
#############################################

load = #LOAD THE PRETRAINED model
load.to(device)

print('Start Feature Extraction')
print()
for epoch in range(1, 2):
    dual.eval()
    load.eval()
    Feature_Y = np.empty([0, 1]) # For LABEL 
    Feature_X = [] # For REPRESENTATION 
    i=0
    for data1 in Ext_Loader:
        images, label = data1

        labels_np = label.numpy()
        Feature_Y = np.concatenate((Feature_Y, labels_np), axis=None)

        # General
        images = images.to(device)
        output, features = load(images)

        features = features.cpu().detach().numpy()
        features = features.flatten()
        Feature_X.append(features)

Feature_X = np.array(Feature_X) #THE FINAL LATENT REPRESENTATIONS OF LOADER (Ext_Loader)


##SAVE Feature_X and Feature_Y separately, This is for the baseline network



######Feature EXTRACTION of ADDITIONAL NETWORKS

load = #LOAD THE PRETRAINED NETWORK (1st level)
load.to(device)
dual = #LOAD 2nd and 3rd model separately.
###Note that, you can only adjust a specific model (2nd level nework or 3rd level network) 
### You have to extract both of them separately. 


print('Start Feature Extraction')
print()
for epoch in range(1, 2):
    load.eval()
    Feature_Y = np.empty([0, 1])
    Feature_X = []
    i=0
    for data1 in Ext_Loader:
        images, label = data1

        labels_np = label.numpy()
        Feature_Y = np.concatenate((Feature_Y, labels_np), axis=None)

        #final
        images = images.to(device)
        output__, features__ = load(images)
        output__ = output__.to(device)
        output, features = dual(output__)

        features = features.cpu().detach().numpy()
        features = features.flatten()
        Feature_X.append(features)
Feature_X = np.array(train_x)

##SAVE Feature_X and Feature_Y separately (This is for additional networks)
