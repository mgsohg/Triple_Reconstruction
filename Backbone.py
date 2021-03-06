from __future__ import division, print_function
import torchvision.datasets as dsets
import torch
import torch.nn as nn
from torchvision.utils import save_image
from torchvision import models, transforms
import numpy as np
import matplotlib.pyplot as plt

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

        self.conv6 = nn.Conv2d(256, 512, kernel_size=7, stride=1)  
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
        self.unpool4 = nn.MaxUnpool2d(kernel_size=2, stride=2) 

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

        '''latent features'''
        x = self.conv6(x)
        latent = x
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
