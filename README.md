# Unsupervised Triple Reconstruction for the Classification of Ultrasound Breast Lesions
___
A pytorch implementation of the paper "*Unsupervised Triple Reconstruction for the Classification of Ultrasound Breast Lesions*' which is under review by a journal of '*Biomedical Signal Processing and Control*', Elsevier.

# Datasets
___

The data  used in this study are publicly available.

**Accessibility:**<br/>
<br/>
**BUSI:**,https://scholar.cu.edu.eg/?q=afahmy/pages/dataset <br/>
**UDIAT:** http://www2.docm.mmu.ac.uk/STAFF/M.Yap/dataset.php<br/>

**If you use this dataset, please cite:**<br/>
<br/>
**BUSI:** Al-Dhabyani W, Gomaa M, Khaled H, Fahmy A. Dataset of breast ultrasound images. Data in Brief. 2020 Feb;28:104863. DOI: 10.1016/j.dib.2019.104863.<br/><br/>
**UDIAT:** Yap, M.H., Pons, G., Marti, J., Ganau, S., Sentis, M., Zwiggelaar, R., Davison, A.K. and Marti, R.(2017), Automated Breast Ultrasound Lesions Detection using Convolutional Neural Networks. IEEE journal of biomedical and health informatics. doi: 10.1109/JBHI.2017.2731873 <br/>


# Triple_Reconstruction
___

Note that, a customized convolutional autoencoder can be found in (Backbone.py)

* Pretraining the first network (pretrain.py)
* Using pretrained model, train two additional networks separately.(2levCAE.py and 3levCAE.py)
* Once the training is done, extracts the features corresponding to trainng and testing sets. (feature_extraction.py)
* Finding the optimal *K* in clustering (In our case, we utilized the method of Silhouette coefficient)
* Fuse the features (Flatten and Concat) and provide it to kmeans clustering with the optimal k (Can be found in kmeans.py)


# Requirements
___
- a Python installation version 3.8.5  
- torch install version 1.7.1 (pytoch.org)
- torchvision install version 0.8.2
- CUDA version 11.4
- the scikit-learn packages
