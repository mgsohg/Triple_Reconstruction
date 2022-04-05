# Unsupervised Triple Reconstruction for the Classification of Ultrasound Breast Lesions
___
A pytorch implementation of the paper "*Unsupervised Triple Reconstruction for the Classification of Ultrasound Breast Lesions*' which is under review by a journal of '*Biomedical Signal Processing and Control*'.



# Triple_Reconstruction
___
The data sets used in this study are publicly available data sets (BUSI:doi:10.1016/j.dib.2019.104863 ,UDIAT: http://www2.docm.mmu.ac.uk/STAFF/M.Yap/dataset.php)
* Pretraining the first network
* Using pretrained model, train two additional networks separately.
* Once the training is done, extracts the features corresponding to trainng and testing sets. 
* Fuse the features (Flatten and Concat) and provide it to kmeans clustering



# Requirements
___
- a Python installation version 3.8.5  
- torch install version 1.7.1 (pytoch.org)
- torchvision install version 0.8.2
- CUDA version 11.4
- the scikit-learn packages
