# Unsupervised Triple Reconstruction for the Classification of Ultrasound Breast Lesions
___
A pytorch implementation of the paper "*Unsupervised Triple Reconstruction for the Classification of Ultrasound Breast Lesions*' which is under review by a journal of '*Biomedical Signal Processing and Control*'.

# Data
___

The data  used in this study are publicly available data sets.

Accessibility:
BUSI:,https://scholar.cu.edu.eg/?q=afahmy/pages/dataset
UDIAT: http://www2.docm.mmu.ac.uk/STAFF/M.Yap/dataset.php

If you use this dataset, please cite:
BUSI: Al-Dhabyani W, Gomaa M, Khaled H, Fahmy A. Dataset of breast ultrasound images. Data in Brief. 2020 Feb;28:104863. DOI: 10.1016/j.dib.2019.104863. UDIAT: Yap, M.H., Pons, G., Marti, J., Ganau, S., Sentis, M., Zwiggelaar, R., Davison, A.K. and Marti, R.(2017), Automated Breast Ultrasound Lesions Detection using Convolutional Neural Networks. IEEE journal of biomedical and health informatics. doi: 10.1109/JBHI.2017.2731873 


# Triple_Reconstruction
___

* Pretraining the first network
* Using pretrained model, train two additional networks separately.
* Once the training is done, extracts the features corresponding to trainng and testing sets. 
* Finding the optimal *K* in clustering (In our case, we utilized the method of Silhouette coefficient)
* Fuse the features (Flatten and Concat) and provide it to kmeans clustering with the optimal k


# Requirements
___
- a Python installation version 3.8.5  
- torch install version 1.7.1 (pytoch.org)
- torchvision install version 0.8.2
- CUDA version 11.4
- the scikit-learn packages
