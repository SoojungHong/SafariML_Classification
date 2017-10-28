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

"""
# on my Mac, this doesn't work. Use code below 
from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original')
mnist 
"""

from sklearn.datasets.mldata import fetch_mldata
import tempfile
test_data_home = tempfile.mkdtemp()

mnist = fetch_mldata('MNIST original', data_home=test_data_home)
mnist 

print(type(mnist))

#------------------------------
# check how data looks like 
X, y = mnist["data"], mnist["target"]
X
X.shape #(70000, 784) - 70000 images, 784 features (28 * 28 pixels)
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
shuffle_index
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

# target vector this classifier
y_train_5 = (y_train == 5)  # True for all 5s, False for all other digits.
y_test_5 = (y_test == 5)

# pick a classifier and train it
from sklearn.linear_model import SGDClassifier
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)
sgd_clf.predict([some_digit])

#------------------------------------------------------------------------------------------------
# implement 'Cross Validation' process which does same thing as Scikit-Learn's cross_val_score()
#------------------------------------------------------------------------------------------------
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

skfolds = StratifiedKFold(n_splits=3, random_state=42)
print(type(skfolds)) #skfolds is class 

for train_index, test_index in skfolds.split(X_train, y_train_5):
    clone_clf = clone(sgd_clf)
    X_train_folds = X_train[train_index]
    X_train_folds
    y_train_folds = y_train_5[train_index]
    y_train_folds
    X_test_fold = X_train[test_index]
    y_test_fold = y_train_5[test_index]

    clone_clf.fit(X_train_folds, y_train_folds)
    y_pred = clone_clf.predict(X_test_fold)
    y_pred
    n_correct = sum(y_pred == y_test_fold)
    n_correct
    print(n_correct / len(y_pred))  # prints 0.9502, 0.96565 and 0.96495
    
#-------------------------------
# claaifier for non 5     

from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_score

class Never5Classifier(BaseEstimator):
    def fit(self, X, y=None):
        pass
    def predict(self, X):
        return np.zeros((len(X), 1), dtype=bool)    
    
never_5_clf = Never5Classifier()
cross_val_score(never_5_clf, X_train, y_train_5, cv=3, scoring="accuracy")    
#--> This demonstrates why accuracy is generally not the preferred performance measure for classifiers, especially when you are dealing with skewed datasets (i.e., when some classes are much more frequent than others).
    