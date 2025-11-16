import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern
from data_generator1 import SampleDataGenerator
import random
from tqdm import tqdm

# ==============================================
# Gaussian Process Regression (kept same structure as Eleanora's code - this bit is unchanged)
# ==============================================
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

        return y_mean, sigma, gp


# ==============================================
# Acquisition functions (NEW CODE ADDED)
# ==============================================
def expected_improvement(mu, sigma, best_y, xi=0.01):
    """Computes the Expected Improvement acquisition value"""
    from scipy.stats import norm

    improvement = mu - best_y - xi
    Z = improvement / sigma
    ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
    ei[sigma == 0.0] = 0.0
    return ei


# ==============================================
# Bayesian Optimization Loop (GP does NOT retrain each iteration)
# ==============================================
def run_bayesian_optimization(initial_data, iterations=20, lenscale=1.5):

    X = initial_data.drop(columns=["yield"])  # input params
    y = initial_data["yield"]                  # target

    # Train initial GP ONCE ONLY (black-box surrogate)
    gp_model = GaussianProcessRegression(X, y)
    _, _, gp = gp_model.regressor(lenscale, X)

    best_y = y.max()
    history = [best_y]

    for step in range(iterations):
        # Sample random candidate points in parameter space
        candidates = []
        for _ in range(500):
            row = []
            for col in X.columns:
                if initial_data[col].dtype == "object":  # categorical
                    row.append(random.choice(initial_data[col].unique()))
                else:
                    low = initial_data[col].min()
                    high = initial_data[col].max()
                    row.append(random.uniform(low, high))
            candidates.append(row)

        candidates = pd.DataFrame(candidates, columns=X.columns)

        # Encode categoricals
        encoded_candidates = candidates.copy()
        for col in encoded_candidates.columns:
            if encoded_candidates[col].dtype == "object":
                le = LabelEncoder()
                encoded_candidates[col] = le.fit_transform(encoded_candidates[col])

        # Predict GP (model no longer gets retrained)
        mu, sigma = gp.predict(encoded_candidates, return_std=True)

        # Compute acquisition values
        ei = expected_improvement(mu, sigma, best_y)

        # Pick best point
        best_idx = np.argmax(ei)
        next_point = candidates.iloc[best_idx]

        # Generate synthetic yield using data generator model
        temp_df = pd.DataFrame([next_point])
        dg = SampleDataGenerator()
        temp_df = dg.depended_correlation(temp_df)
        new_y = temp_df["yield"].values[0]

        # Append to dataset (only for record-keeping)
        X = pd.concat([X, next_point.to_frame().T], ignore_index=True)
        y = pd.concat([y, pd.Series([new_y])], ignore_index=True)

        # ❌ GP is NOT retrained — it stays fixed

        # Track best
        best_y = max(best_y, new_y)
        history.append(best_y)

    return history, X, y
