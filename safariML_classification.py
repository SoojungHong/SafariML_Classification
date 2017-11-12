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
y_train

# target vector this classifier
y_train_5 = (y_train == 5)  # True for all 5s, False for all other digits.
y_train_5

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

#-----------------------------------------------------------------------
# Better way to evaluate the performance : Looking at confusion matrix  
from sklearn.model_selection import cross_val_predict
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)   #return the predictions
y_train_pred

from sklearn.metrics import confusion_matrix
confusion_matrix(y_train_5, y_train_pred)

#confusion_matrix(y_train_5, y_train_perfect_predictions)

#--------------------------
# Precision and Recall 
from sklearn.metrics import precision_score, recall_score
precision_score(y_train_5, y_train_pred) #precision score = True positive / (True positive + False positive)

#-------------------
# F1 Score is Harmonic mean (F1 SCore = True Positive /(True Positive + (False Negative + False Positive )/2))
from sklearn.metrics import f1_score
f1_score(y_train_5, y_train_pred)


#-------------------------------------------------------------------
# you can compute precision and recall for all possible threshold
y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method="decision_function")

from sklearn.metrics import precision_recall_curve
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)
def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision") #precisions[:-1] means everything except the last item
    plt.plot(thresholds, recalls[:-1], "g--", label="Recall" )
    plt.xlabel("Threshold")
    plt.legend(loc="center left")
    plt.ylim([0, 1]) #ylim is getting y limits in axis 
    
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.show()    


#----------------------------------------------------------------------------------
# ROC (Receiver Operating Curve) - plot true positive rate vs. false positive rate
# ROC plot sensitivity(recall) versus 1-specificity
# specificity = true negative rate
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)

def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

plot_roc_curve(fpr, tpr)
plt.show()    

#--------------------------------------------------------------------
# ROC AUC (Receiver Operating Characteristics Area Under the Curve)
from sklearn.metrics import roc_auc_score
roc_auc_score(y_train_5, y_scores)
f1_score(y_train_5, y_train_pred)


#------------------------------------------------------------------------------------
# comparision of Random Forest Classifier and Stochastic Gradient Descent classifier
# using ROC curve and ROC AUC curve
from sklearn.ensemble import RandomForestClassifier
forest_clf = RandomForestClassifier(random_state=42)
forest_clf
y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3, method="predict_proba")
y_scores_forest = y_probas_forest[:,1] # score = proba of positive class
fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5, y_scores_forest)
plt.plot(fpr, tpr, "b:", label="SGD")
plot_roc_curve(fpr_forest, tpr_forest, "Random Forest")
plt.legend(loc="lower right")
plt.show()

roc_auc_score(y_train_5, y_scores_forest)


#----------------------------------------
# multiclass (multinomial) classifier 
y_train #y_train is target class from 0 to 9
y_train_5
sgd_clf.fit(X_train, y_train) # attention! y_train, not y_train_5
sgd_clf.predict([some_digit])

some_digit_scores = sgd_clf.decision_function([some_digit])
some_digit_scores 

np.argmax(some_digit_scores)
sgd_clf.classes_
sgd_clf.classes_[5]

#----------------------------------------------------------------------
# using Scikit learn  one vs one classifier , one vs rest classifier 
from sklearn.multiclass import OneVsOneClassifier
ovo_clf = OneVsOneClassifier(SGDClassifier(random_state = 42))
ovo_clf.fit(X_train, y_train)
ovo_clf.predict([some_digit])
len(ovo_clf.estimators_)
ovo_clf.estimators_

forest_clf.fit(X_train, y_train)
forest_clf.predict([some_digit])
forest_clf.predict_proba([some_digit])
cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy")

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring="accuracy")


#------------------------------------
# Error Analysis 

# First, look at confusion matrix
y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)
y_train_pred
conf_mx = confusion_matrix(y_train, y_train_pred)
conf_mx
plt.matshow(conf_mx, cmap=plt.cm.gray)
plt.show()