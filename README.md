<img src="https://github.com/quartermaine/Master-Thesis/blob/main/LiU.jpeg" width="150" height="150"/>

# Master Thesis LiU 

<figure>
  <img src="https://github.com/quartermaine/Master-Thesis/blob/main/thesis%20images/Data/degree_binarize.png" width="300" height="200"/>  <figcaption>Fig.1 - Trulli, Puglia, Italy.</figcaption> <img src="https://github.com/quartermaine/Master-Thesis/blob/main/thesis%20images/Data/FMRI.jpg" width="300" height="200"/> 
<img src="https://github.com/quartermaine/Master-Thesis/blob/main/thesis%20images/Theory/NSjRyPyygz-derp.JPG" width="400" height ="300"/>0 <img src="https://github.com/quartermaine/Master-Thesis/blob/main/thesis%20images/Data/dual_regression.png" width="400" height ="300"/> 
 </figure>


This repository contains the code and visualizations of my master thesis in brain disease classification using multi-channel 3D convolutional neural networks. The thesis can be found [here](https://www.diva-portal.org/smash/record.jsf?dswid=1015&pid=diva2%3A1538345&c=1&searchType=SIMPLE&language=en&query=andreas+christopoulos+charitos&af=%5B%5D&aq=%5B%5B%5D%5D&aq2=%5B%5B%5D%5D&aqe=%5B%5D&noOfRows=50&sortOrder=author_sort_asc&sortOrder2=title_sort_asc&onlyFullText=false&sf=all). :octocat:

# Abstract

  Functional magnetic resonance imaging (fMRI) technology has been used in the investigation of human brain functionality and assist in brain disease diagnosis. While fMRIcan be used to model both spatial and temporal brain functionality, the analysis of the fMRIimages and the discovery of patterns for certain brain diseases is still a challenging task inmedical imaging.
  
  Deep learning has been used more and more in medical field in an effort to furtherimprove disease diagnosis due to its effectiveness in discovering high-level features in im-ages. Convolutional neural networks (CNNs) is a class of deep learning algorithm thathave been successfully used in medical imaging and extract spatial hierarchical features. The application of CNNs in fMRI and the extraction of brain functional patterns is an openfield for research. This project focuses on how fMRIs can be used to improve Autism Spec-trum Disorders (ASD) detection and diagnosis with 3D resting-state functional MRI (rs-fMRI) images. ASDs are a range of neurodevelopment brain diseases that mostly affectsocial function. Some of the symptoms include social and communicating difficulties, andalso  restricted and repetitive behaviors. The symptoms appear on early childhood and tend to develop in time thus an early diagnosis is required.
  
  Finding a proper model for identifying between ASD and healthy subject is a challenging task and involves a lot of hyper-parameter tuning. In this project a grid search approach is followed in the quest of the optimal CNN architecture. Additionally, regularization and augmentation techniques are implemented in an effort to further improve the models performance.

# Thesis goal

In the present study the main aim is to use DL algorithms and develop a system than candistinguish between ASD and CON subjects using resting state fMRI data. The data used in this project is the [ABIDE](http://preprocessed-connectomes-project.org/abide/index.html) fMRI brain volume dataset which is freely available. 

# Research questions

- RQ1: What combination of derivatives gives the highest classification accuracy?  
- RQ2: Does data augmentation improve 3D CNN performance
- RQ3: What are the most important hyper-parameters for the classification accu-racy? More specifically we wish to know how many layers and filters to use.
- RQ4:  Is the classification uncertainty reduced when training the 3D CNN withaugmentation, compared to when no augmentation is used?

# Download the ABIDE dataset 

* [data_download_one_derivative.sh](https://github.com/quartermaine/Master-Thesis/blob/main/download%20data/data_download_one_derivative.sh)

A bash scipt to download a single derivative from ABIDE dataset

``` bash
./data_download_one_derivative.sh
```

* [download_data.sh](https://github.com/quartermaine/Master-Thesis/blob/main/download%20data/download_data.sh)

A bash scipt to download a list of derivatives as specified by ```listOfDerivatives="alff degree_binarize degree_weighted dual_regression eigenvector_binarize eigenvector_weighted falff lfcd reho vmhc"```

``` bash
./data_download_data.sh
```

Inside the scripts the parameters pipeline and strategy can be set. For a list of the available pipelines and strategies refer [here](https://github.com/preprocessed-connectomes-project/abide/blob/master/download_abide_preproc_guide.txt)  

# Thesis train scripts 

The scripts used in the present thesis are 3. Two scipts for combining different derivatives to multichanell volumes with/without augmentation and a script to train the CNN. 


* [write_TfRecords_parser_no_aug]()

A python scipt to combine different derivatives without augmentation.

```python
$python write_TfRecords_parser_no_aug.py -h # for help
```

* [write_TfRecords_parser_aug]()

A python scipt to combine different derivatives and perform augmentations using multiprocessing for speedup.

```python
$python write_TfRecords_parser_aug.py -h # for help
```

The data from the above two sciprts are stored in TfRecords format.

* [train_keras_augmentation_v2]()

This scipt performs the grid search training using the TfRecods that created with one of the previous scipts. The scipt takes no arguments but inside the code   different parameters can be set for the grid search (#convolutional layers, #dense nodes, e.t.c.) 

```python
$python train_keras_augmentation_v2.py 
```

# File structure 

The directory and file structure is shown below.

```
+-- Code
|   +-- write_TfRecords_parser_aug.py
|   +-- write_TfRecords_parser_no_aug.py
|              .
|              .
|              .
|   +-- train_keras_augmentation_v2.py
+-- Train
|   +-- DATA 
|         +-- alff
|              .
|              .
|              .
|         +-- vmhf
|   +-- TfRecords
|         +-- dual_regression_aug    
|         +-- falff_no_aug
|              .
|              .
|              .
|         +-- alff_degree_binarize_..._reho_vmhc_aug
```
