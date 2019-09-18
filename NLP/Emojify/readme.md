# Emojify
---
This project implements "Emojify" with Pytorch.

- Input: Sentences 
- Output: Emoji (cast as numerical labels)🤔

For example:
Food is life 🍴

## Acknowledgement
Some ideas and the structure of the neural network come from [Coursera Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning).

The dataset can be download from [here](https://drive.google.com/drive/folders/1vXgzjhALvH981cNYZwlQ1wZJ_NE_Xd44?usp=sharing).

Preview dataset:
- never talk to me again 😞
- I am proud of your achievements 😄
- It is the worst day in my life 😞
- Miss you so much ❤️
- food is life 🍴

## Content
- [Step-by-step notebook: emoji_pytorch.ipynb](Emojify_pytorch.ipynb)

- [Python script: emoji_script.py](emoji_script.py) 

Usage: `python emoji_script.py`

## Dependencies
- numpy
- pytorch
- pandas
- tqdm
- emoji

## Result
Training accuracy: 100%
Test accuracy: 91.07%


<img src="example.png" alt="drawing" width="600"/>
