import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn

#og
train_x1 = #load the first representations
train_x2 = #load the second representations
train_x3 = #load the third representations
train_y = #Load the label

'''You can use any tran_y between first, second, and third network since we extracted features without shuffling.'''

# Flatten && Concat for training set
train_x1 = train_x1.flatten()
train_x2 = train_x2.flatten()

mid_train_ = np.concatenate([train_x1,train_x2])
mid_trainx_ = np.reshape(mid_train_,(N,1024)) ## (A,B) A: total number of N, B: Volume of feature *2

Final_train = np.concatenate([train_x3,mid_train_])
Final_train = np.reshape(Final_train,(N,1536)) ## (A,B) A: total number of N, B: Volume of feature *3

Train_x, Train_y = shuffle(Final_train ,train_y) #Shuffling


# Flatten && Concat for testing set

test_x1 = #load the first representations
test_x2 = #load the second representations
test_x3 = #load the third representations
test_y = #load the  label

test_x1= test_x1.flatten()
test_x2 = test_x2.flatten()

mid_test = np.concatenate([test_x1,test_x2])
mid_testx = np.reshape(mid_test,(N,1024))

mid_testx = mid_testx.flatten()
test_x3 = test_x3.flatten()
Final_test = np.concatenate([mid_testx,test_x3])
Final_test = np.reshape(Final_test,(N,1536))

Test_x, Test_y = Final_test, test_y


##Kmeans

from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix

kmeans = KMeans(n_clusters=K) #Optimal K
kmeans.fit(Train_x)
y_pred = kmeans.predict(Test_x)
