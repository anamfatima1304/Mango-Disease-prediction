
# Mango Disease Prediction

Disese prediction using deep neural networks help in identifying the diseases using various images. This project uses the images collected from the Four mango orchards of Bangladesh to create a model that can identify 8 different types of diseases of Mango. 

## Technologies 
The **Mango Disease Prediction** follows a step-by-step procedure to build an efficient model to predict mango diseases. The dataset is cleaned, visualized, augmented and predicted in a proper way. The following technologies are used to make the project possible:

- **Python**
- **Numpy and Pandas for data cleaning**
- **Matplotlib for data visualization**
- **Tensorflow**
- **Jupyter notebook and visual studio code as IDE**

The **MangoDiseasePrediction
.ipynb** implements the deep learning algorithm on the given dataset using the keras. This model predicts the diseases by analyzing the leaves and gives an accuracy of **91%**.

The **Prediction.ipynb** imports the saved model and makes prediction on the given Image. The image is given by specifying its path.

## Deployment

To use the project, run the following command in the folder where you want to get the project.

```bash
  https://github.com/anamfatima1304/Mango-Disease-prediction.git
```
Now play around with the model and enjoy.

## Installation

To run this project, you should have python istalled on your computer. You might also need an IDE like VSCode or Jupyter Notebook that could run the code. Also import the following packages to run the code smoothly.

```bash
    import tensorflow as tf
    import numpy as np
    from tensorflow.keras import models, layers
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
    import os
    import pickle
```


## Dataset

The dataset uses 4000 240x320 images that are divided into 9 categories. 8 categories describe 8 different disease that a mango tree encounter while the 9th one identifies the healthy leaves. The dataset is downloaded from [Kaggle](https://www.kaggle.com/datasets/aryashah2k/mango-leaf-disease-dataset) that predicts various diseases which are Anthracnose, Bacterial Canker, Cutting Weevil, Die Back, Gall Midge, Powdery Mildew, and Sooty Mould. I, whole heartedly, pay special thanks to Ali, Sawkat; Ibrahim, Muhammad ; Ahmed, Sarder Iftekhar ; Nadim, Md. ; Mizanur, Mizanur Rahman; Shejunti, Maria Mehjabin ; Jabid, Taskeed (2022), “MangoLeafBD Dataset”, Mendeley Data, V1, doi: 10.17632/hxsnvwty3r.1 for creating such an amazing database. 
## Contributing

Contributions are always welcome!

Find out an Error or give any suggestion to improve the project.

If you want to add any new features, make a pull request.

ThankYou for your attention.
