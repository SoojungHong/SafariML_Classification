# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 16:44:30 2017

@author: a613274
@about: chapter 4. training model 
    
"""

import numpy as np
X = 2 * np.random.rand(100, 1)
X
y = 4 + 3 * X + np.random.randn(100,1)
y

#-----------------------------------
# closed form approach
# find theta using normal equation
X_b = np.c_[np.ones((100, 1)), X]  # add x0 = 1 to each instance
X_b
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y) # approximation of theta = transpose of (X transpose dot product of X) dot product of X transpose (dot product of y)
theta_best

X_new = np.array([[0], [2]])
X_new

X_new_b = np.c_[np.ones((2,1)), X_new] #add x0 = 1 to each instance, np.ones((2,1)) means array([[ 1.],[1.]])
X_new_b
y_predict = X_new_b.dot(theta_best)
y_predict

#-----------------
# plot the data
from matplotlib import pyplot as plt
plt.plot(X_new, y_predict, "r-")
plt.plot(X, y, "b.")
plt.axis([0,2,0,15])
plt.show()

#-----------------
# using model
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)
lin_reg.intercept_, lin_reg.coef_
lin_reg.predict(X_new)

#--------------------------------------
# Gradient Descent and Learning rate
eta = 0.1  #learning rate
n_iterations = 1000 
m = 100 

theta = np.random.randn(2,1) #randon initialization 

for iteration in range(n_iterations):
    gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
    theta = theta - eta * gradients 

theta   

#-------------------------------
# Stochastic Gradient Descent 
n_epochs = 50
t0, t1 = 5, 50 #learning schedule parameters 

def learning_schedule(t): 
    return t0/(t + t1)

theta = np.random.randn(2, 1) #random initialization 
theta
for epoch in range(n_epochs):
    for i in range(m):
        random_index = np.random.randint(m)
        xi = X_b[random_index:random_index+1]
        yi = y[random_index:random_index+1]
        gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
        eta = learning_schedule(epoch * m + i)
        theta = theta - eta  * gradients 

theta     


#----------------------------------------
# Linear regression using SGDRegressor   
from sklearn.linear_model import SGDRegressor
sgd_reg = SGDRegressor(n_iter=50, penalty=None, eta0=0.1)
sgd_reg.fit(X, y.ravel())  #ravel return the continuous flattened array 
sgd_reg.intercept_
sgd_reg.coef_