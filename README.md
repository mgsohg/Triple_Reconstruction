# Unsupervised Learning Method via Triple Reconstruction for the Classification of Ultrasound Breast Lesions
___
A pytorch implementation of the paper "*Unsupervised Learning Method via Triple Reconstruction for the Classification of Ultrasound Breast Lesions*' which is accpeted by a journal of '*Biomedical Signal Processing and Control*'.

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

Note that, a customized backbone architecture can be found in "Backbone.py"

* Pretrain the initial network in a generic way "pretrain.py"
* Feeding the first level reconstructed output to the second network and thrid network via pretrained model, separately."2levCAE.py and 3levCAE.py", respectively.
* Once the training is done, extract the individual latent representation corresponding to each instance. (feature_extraction.py)
* Finding the optimal *K* in clustering (In our case, we utilized the method of Silhouette coefficient)
* Generate the mutual information, and feed to kmeans clustering with the optimal k (Can be found in kmeans.py) in order to obtain the predicted labels.

Also, it is worth mentioning that, the purpose of generating label information is to measure the discriminant performance by comparing with prediction labels, instead of utilizing it during the training phase.



# Requirements
___
- a Python installation version 3.8.5  
- torch install version 1.7.1 (pytoch.org)
- torchvision install version 0.8.2
- CUDA version 11.4
- the scikit-learn packages
