# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 16:59:21 2017

@author: a613274
@Safari ML, Chapter 3 : Classification
https://www.safaribooksonline.com/library/view/hands-on-machine-learning/9781491962282/ch03.html
"""

#-------------------------------------------
# Fetch data
#-------------------------------------------
from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original')
mnist 
print(type(mnist))

#------------------------------
# check how data looks like 
X, y = mnist["data"], mnist["target"]
X
X.shape
y
y.shape

#---------------------------
# take a peek at one digit 
#%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt

some_digit = X[36000]
some_digit
some_digit_image = some_digit.reshape(28, 28)
some_digit_image

plt.imshow(some_digit_image, cmap = matplotlib.cm.binary,
           interpolation="nearest")
plt.axis("off")
plt.show()
y[36000]


#--------------------------------------------------------------------------------------------------------------------------------
# Divide dataset into Test and Train 
# The MNIST dataset is actually already split into a training set (the first 60,000 images) and a test set (the last 10,000 images)
#-------------------------------------------------------------------------------------------------------------------------------
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
X_train
import numpy as np

#-------------------------------------------------------------------
# shuffle index so, the cross validation folds will not be similar
shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]