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

X_new_b = np.c_[np.ones((2,1)), X_new] #add x0 = 1 to each instance
X_new_b
y_predict = X_new_b.dot(theta_best)
y_predict

#-----------------
# plot the data
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