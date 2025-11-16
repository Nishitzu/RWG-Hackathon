import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern

#Creating a class for easier manipulation and further expansion
class GaussianProcessRegression():
    def __init__(self, train_X, train_Y):
        #Adding the train sets to the constructor
        self.train_X = train_X
        self.train_Y = train_Y

    #Creating a modular method for model training and test, it can be split
    def regressor(self, lenscale, test_X):

        #Chose RBF (Radial Basis Function) for kernel as it is the favorite and most used, it can be changed to Matern,
        #which is an approximate RBF for less smooth data
        #this kernel requires a length_scale argument, which determines how far from the actual data the model
        #can try to extrapolate the pattern (larger should mean less overfitting)
        kernel = 1.0 * RBF(length_scale=lenscale)

        #Instantiated the Gaussian Process Regressor with the chosen kernel
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)

        gp.fit(self.train_X, self.train_Y)

        #This step can be separated, I am rushing to be honest
        y_mean, sigma = gp.predict(test_X, return_std=True)

        return y_mean, sigma






