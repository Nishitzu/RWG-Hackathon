import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from data_generator1 import SampleDataGenerator
from tqdm import tqdm

###############################################
# FILE: gp_train_loss_plot.py
# Purpose: Split data, train GP ONCE, plot loss
###############################################

def train_gp_and_plot_loss():
    # 1. Generate synthetic dataset
    gen = SampleDataGenerator()
    df = gen.syntetic_data_gen()
    df = gen.depended_correlation(df)

    # 2. Encode categorical variables
    cat_cols = ["species", "feeding_regime", "feedstock"]
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    # 3. Train-test split
    X = df.drop(columns=["yield"])
    y = df["yield"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # 4. Track GP training loss across optimizer restarts
    losses = []
    print("\nTraining GP across 10 restarts...\n")

    for restart in tqdm(range(10), desc="GP Restarts"):
        # RANDOMIZE the initial length-scale so each restart is different
        random_ls = np.random.uniform(0.1, 5.0)

        kernel = RBF(length_scale=random_ls)

        # Enable real kernel hyperparameter optimization
        gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=5,   # GP now optimizes hyperparameters
            normalize_y=True
        )

        gp.fit(X_train, y_train)

        # Compute training loss (NLML)
        loss = -gp.log_marginal_likelihood(gp.kernel_.theta)
        losses.append(loss)

    # 5. Plot training loss
    losses = np.array(losses)
    running_best = np.minimum.accumulate(losses)

    plt.figure(figsize=(8, 5))
    plt.plot(losses, marker='o', label="Loss per restart")
    plt.plot(running_best, marker='x', linestyle='--', label="Best loss so far")
    plt.title("Gaussian Process Training Loss (NLML) Across Restarts")
    plt.xlabel("Optimizer Restart Number")
    plt.ylabel("Loss (Negative Log Marginal Likelihood)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return gp

# If run directly, execute the training + plot
def main():
    train_gp_and_plot_loss()

if __name__ == "__main__":
    main()
