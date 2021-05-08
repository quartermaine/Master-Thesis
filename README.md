# Master Thesis LiU


<img src="https://github.com/quartermaine/Master-Thesis/blob/main/thesis%20images/Data/degree_binarize.png" width="400" height="300"/> <img src="https://github.com/quartermaine/Master-Thesis/blob/main/thesis%20images/Data/FMRI.jpg" width="400" height="300"/> 
<img src="https://github.com/quartermaine/Master-Thesis/blob/main/thesis%20images/Theory/NSjRyPyygz-derp.JPG" width="400" height ="300"/>0 <img src="https://github.com/quartermaine/Master-Thesis/blob/main/thesis%20images/Data/dual_regression.png" width="400" height ="300"/>


This repository contains the code and visualizations of my master thesis in brain disease classification using multi-channel 3D convolutional neural networks. The thesis can be found [here](https://www.diva-portal.org/smash/record.jsf?dswid=1015&pid=diva2%3A1538345&c=1&searchType=SIMPLE&language=en&query=andreas+christopoulos+charitos&af=%5B%5D&aq=%5B%5B%5D%5D&aq2=%5B%5B%5D%5D&aqe=%5B%5D&noOfRows=50&sortOrder=author_sort_asc&sortOrder2=title_sort_asc&onlyFullText=false&sf=all). :octocat:

# Abstract

  Functional magnetic resonance imaging (fMRI) technology has been used in the investigation of human brain functionality and assist in brain disease diagnosis. While fMRIcan be used to model both spatial and temporal brain functionality, the analysis of the fMRIimages and the discovery of patterns for certain brain diseases is still a challenging task inmedical imaging.
  
  Deep learning has been used more and more in medical field in an effort to furtherimprove disease diagnosis due to its effectiveness in discovering high-level features in im-ages. Convolutional neural networks (CNNs) is a class of deep learning algorithm thathave been successfully used in medical imaging and extract spatial hierarchical features. The application of CNNs in fMRI and the extraction of brain functional patterns is an openfield for research. This project focuses on how fMRIs can be used to improve Autism Spec-trum Disorders (ASD) detection and diagnosis with 3D resting-state functional MRI (rs-fMRI) images. ASDs are a range of neurodevelopment brain diseases that mostly affectsocial function. Some of the symptoms include social and communicating difficulties, andalso  restricted and repetitive behaviors. The symptoms appear on early childhood and tend to develop in time thus an early diagnosis is required.
  
  Finding a proper model for identifying between ASD and healthy subject is a challenging task and involves a lot of hyper-parameter tuning. In this project a grid search approach is followed in the quest of the optimal CNN architecture. Additionally, regularization and augmentation techniques are implemented in an effort to further improve the models performance.

# Thesis goal

In the present study the main aim is to use DL algorithms and develop a system than candistinguish between ASD and CON subjects using resting state fMRI data. 

# Research questions

- RQ1: What combination of derivatives gives the highest classification accuracy?  
- RQ2: Does data augmentation improve 3D CNN performance
- RQ3: What are the most important hyper-parameters for the classification accu-racy? More specifically we wish to know how many layers and filters to use.
- RQ4:  Is the classification uncertainty reduced when training the 3D CNN withaugmentation, compared to when no augmentation is used?

# Download the ABIDE dataset 


``` bash
./download.sh
```


# Thesis train scripts 

The scripts used in the present thesis 



```python
from mazeexplorer import MazeExplorer

train_env = MazeExplorer(number_maps=1,
                         size=(15, 15),
                         random_spawn=True,
                         random_textures=False,
                         keys=6)
              
test_env = MazeExplorer(number_maps=1,
                        size=(15, 15),
                        random_spawn=True,
                        random_textures=False,
                        keys=6)

# training
for _ in range(1000):
    obs, rewards, dones, info = train_env.step(train_env.action_space.sample())
    
    
# testing
for _ in range(1000):
    obs, rewards, dones, info = test_env.step(test_env.action_space.sample())
```


