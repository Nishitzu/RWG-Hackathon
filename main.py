from gp_and_aq_func import GaussianProcessRegression
from data_generator import SampleDataGenerator
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt

if __name__ == "__main__":

    #The code is fairly simple
    #I hope I haven't botched anything
    data_gen = SampleDataGenerator()

    synt_data = data_gen.syntetic_data_gen()
    synt_data_yield = data_gen.depended_correlation(synt_data)

    print(synt_data_yield.head(50))
    synt_data_yield.to_csv("sample_syntetic_data.xlsx")

    le = LabelEncoder()
    label_carb = le.fit_transform(synt_data["carb_source"])
    print(label_carb)
    label_feeding = le.fit_transform(synt_data["feeding_regime"])
    print(label_carb)

    synt_data["carb_source"] = label_carb
    synt_data["feeding_regime"] = label_feeding

    X = synt_data.loc[:, "ph":"feeding_regime"]
    y = synt_data["yield"]

    print(X)
    print(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=0)

    model = GaussianProcessRegression(X_train, y_train)

    #Couldn't make it in time to implement the acquisition function
    #The acquisition function is the one responsible for optimization
    #We have 3 choices, basically: Probability of Improvement, Expected Improvement, GP Upper Confidence Bound
    #I was honestly thinking about implementing GP UCB because it should be the one which maximises output
    lenscale = 1.5

    y_mean, sigma = model.regressor(lenscale, X_test)

    print(y_mean)
    print(sigma)

    plt.plot(X["ph"], y, label=r"$f(x) = x \sin(x)$", linestyle="dotted")
    plt.scatter(X_train["ph"], y_train, label="Observations")
    plt.plot(X_test["ph"], y_mean, label="Mean prediction")

    plt.legend()
    plt.xlabel("$x$")
    plt.ylabel("$f(x)$")
    _ = plt.title("Gaussian process regression on noise-free dataset")
    plt.show()
