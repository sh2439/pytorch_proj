### Image Folder Description
This folder contains image examples and the format of the image folder.

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
