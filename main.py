from gp_and_aq_func import GaussianProcessRegression, run_bayesian_optimization
from data_generator1 import SampleDataGenerator
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt

if __name__ == "__main__":

    #The code is fairly simple
    #I hope I haven't botched anything
    data_gen = SampleDataGenerator()

    # Generate synthetic data
    synt_data = data_gen.syntetic_data_gen()
    synt_data_yield = data_gen.depended_correlation(synt_data)

    print(synt_data_yield.head(50))
    synt_data_yield.to_csv("sample_synthetic_data.xlsx")

    # Encode categorical columns
    cat_cols = ["species", "feeding_regime", "feedstock"]
    for col in cat_cols:
        le = LabelEncoder()
        synt_data[col] = le.fit_transform(synt_data[col])

    # Inputs: all columns except yield
    X = synt_data.drop(columns=["yield"])
    y = synt_data["yield"]

    print(X.head())
    print(y.head())

    # Split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=0
    )

    # Train GP model (as before)
    model = GaussianProcessRegression(X_train, y_train)

    #Couldn't make it in time to implement the acquisition function
    #The acquisition function is the one responsible for optimization
    #We have 3 choices, basically: Probability of Improvement, Expected Improvement, GP Upper Confidence Bound
    #I was honestly thinking about implementing GP UCB because it should be the one which maximises output
    lenscale = 1.5

    y_mean, sigma, gp = model.regressor(lenscale, X_test)

    print("GP Predictions (mean):")
    print(y_mean)
    print("GP Uncertainty (sigma):")
    print(sigma)

    # ---- NEW: BAYESIAN OPTIMIZATION ----
    print("\nRunning Bayesian Optimization...\n")
    history, X_final, y_final = run_bayesian_optimization(synt_data_yield, iterations=20)

    print("BO Best yield:", max(history))
    print("BO history:", history)

    # ---- PLOT GP PREDICTION FOR ONE PARAMETER (just to keep her plot) ----
    plt.scatter(X_train["ph"], y_train, label="Observations (train)")
    plt.plot(X_test["ph"], y_mean, label="GP mean prediction")
    plt.xlabel("$x$")
    plt.ylabel("$f(x)$")
    plt.title("Gaussian Process Regression on Synthetic Dataset")
    plt.legend()
    plt.show()
