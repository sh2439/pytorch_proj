## Gender Classification from Transfer Learning

### Introduction
Gender classification has been used in lots of maching learning applications. Classifying gender of a person can be simple for humans but it's still an active problem in modern computer vision. State-of-the-art face detection algorithms have reached high accuracy on available benchmark datasets. This project implements transfer learning on gender classification with the help of pretrained vgg-16 face descriptor model.

The pretrained weights can be downloaded from [Pretrained weights](https://drive.google.com/open?id=1gFwEhuTMfdy5jLdYOx_x38MXTly4GurQ), which was extracted from original VGG Face Descriptor Caffe model (The original model can be downloaded from [here](http://www.robots.ox.ac.uk/~vgg/software/vgg_face/)).

### Dataset
The dataset can be downloaded from [Face Image Project](https://talhassner.github.io/home/projects/Adience/Adience-data.html#agegender). The basic structure of the face dataset folder (default name: combined):

       ├── aligned  <-- 29,437 train data
       |   ├── 01_F <-- This subfolder contains the images with gender 'F' and age `01`.
       |   ├── 01_M
       |   ├── 02_F
       |   ├── 02_M
       |   └── ...
       └── valid    <-- 3,681 test data
           ├── 01_F 
           ├── 01_M
           ├── 02_F
           └── ...
          

The size of each image is `128*128*3` with color channels `RGB`. 

<center>Example image:</center>  
<center><img src="image_folder/landmark_aligned_face.205.9615551622_19818245ec_o.jpg" alt="drawing" width="300"/></center>

### Dependencies
- numpy
- configargparse
- argparse
- pytorch
- torchvision
- tqdm (progress bar)

### Usage
`python train.py -c gender_vgg.config`


