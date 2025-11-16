import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ============================================
# 1. DEFINE PARAMETER SPACE + TRUE YIELD MODEL
# ============================================

# Parameter ranges (4D input space)
PARAM_RANGES = {
    "ph": (6.8, 7.2),
    "temp": (27.0, 37.0),
    "s2_carb_conc": (20.0, 60.0),
    "s2_agitation": (300.0, 700.0),
}

def quad_peak(x, x_opt, width):
    """Simple quadratic peak: returns value in [0, 1] (clipped)."""
    val = 1.0 - ((x - x_opt) / width) ** 2
    return np.maximum(0.0, val)


def true_yield_function(ph, temp, s2_carb, s2_agit):
    """
    This is the 'ground truth' PHA yield model.
    Smooth, GP-friendly, with a clear optimum in 4D space.
    Output is scaled to roughly between ~0.2 and 2.0.
    """

    # Quadratic peak for pH around 7.0
    score_ph = quad_peak(ph, 7.0, 0.2)

    # Temperature: optimum around 32°C
    score_temp = quad_peak(temp, 32.0, 5.0)

    # Stage 2 carbon: optimum around 40 g/L
    score_carb = quad_peak(s2_carb, 40.0, 15.0)

    # Stage 2 agitation: optimum around 500 rpm
    score_agit = quad_peak(s2_agit, 500.0, 150.0)

    # Weighted combination
    score = (
        0.30 * score_ph +
        0.25 * score_temp +
        0.25 * score_carb +
        0.20 * score_agit
    )

    # Add a tiny bit of noise (simulating experimental noise)
    noise = np.random.normal(loc=0.0, scale=0.02, size=np.shape(score))

    score_noisy = np.clip(score + noise, 0.0, 1.0)

    # Map score in [0,1] to yield in [0.2, 2.0]
    yield_val = 0.2 + score_noisy * (2.0 - 0.2)
    return yield_val


# ============================================
# 2. SYNTHETIC DATA GENERATION
# ============================================

def sample_uniform_params(n_samples):
    """Sample n_samples points uniformly from the 4D parameter ranges."""
    ph = np.random.uniform(*PARAM_RANGES["ph"], size=n_samples)
    temp = np.random.uniform(*PARAM_RANGES["temp"], size=n_samples)
    s2_carb = np.random.uniform(*PARAM_RANGES["s2_carb_conc"], size=n_samples)
    s2_agit = np.random.uniform(*PARAM_RANGES["s2_agitation"], size=n_samples)

    X = np.vstack([ph, temp, s2_carb, s2_agit]).T
    return X


def generate_dataset(n_samples=2000):
    """Generate a labelled dataset (X, y) using the true yield function."""
    X = sample_uniform_params(n_samples)
    ph, temp, s2_carb, s2_agit = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
    y = true_yield_function(ph, temp, s2_carb, s2_agit)
    return X, y


# ============================================
# 3. TRAIN GP MODEL AS SURROGATE
# ============================================

def train_gp_model(X, y):
    """
    Train a Gaussian Process model on the synthetic dataset.
    Returns the trained GP and the scaler used for X.
    """
    # Scale inputs for better GP behaviour
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Reasonable kernel: constant * RBF
    kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=np.ones(X.shape[1]),
                                       length_scale_bounds=(1e-2, 1e2))

    gp = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=5,
        alpha=1e-4,          # small noise term
        normalize_y=True,
        random_state=0
    )

    gp.fit(X_scaled, y)

    print("\nTrained GP kernel:", gp.kernel_)
    return gp, scaler


# ============================================
# 4. ACQUISITION FUNCTION: EXPECTED IMPROVEMENT
# ============================================

from scipy.stats import norm

def expected_improvement(mu, sigma, best_y, xi=0.01):
    """
    Expected Improvement acquisition function.
    mu, sigma: predictions from GP
    best_y: best observed (or predicted) yield so far
    """
    sigma = np.maximum(sigma, 1e-9)  # avoid division by zero
    improvement = mu - best_y - xi
    Z = improvement / sigma
    ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
    ei[sigma <= 0.0] = 0.0
    return ei


# ============================================
# 5. BAYESIAN-STYLE OPTIMIZATION LOOP (FIXED GP)
# ============================================

def bayesian_optimization_with_fixed_gp(gp, scaler, n_iterations=20, n_candidates=1000):
    """
    BO-style loop:
    - GP is treated as a FIXED black-box model (no retraining).
    - At each iteration:
        * Sample candidate points in 4D
        * Use GP to predict mu, sigma
        * Use EI to pick the most promising candidate
        * Record the predicted yield

    Returns:
        history_y : list of best predicted yield after each iteration
        chosen_points : list of chosen input points (4D)
    """

    history_y = []
    chosen_points = []

    best_y = -np.inf

    for it in range(n_iterations):
        # Sample candidate inputs
        X_cand = sample_uniform_params(n_candidates)
        X_cand_scaled = scaler.transform(X_cand)

        # GP predictions
        mu, sigma = gp.predict(X_cand_scaled, return_std=True)

        # Compute acquisition
        ei = expected_improvement(mu, sigma, best_y if best_y != -np.inf else mu.max())

        # Pick best candidate according to EI
        best_idx = np.argmax(ei)
        x_best = X_cand[best_idx]
        y_best = mu[best_idx]

        chosen_points.append(x_best)
        best_y = max(best_y, y_best)
        history_y.append(y_best)

        print(f"Iteration {it+1:2d}: chosen params = {x_best}, GP predicted yield = {y_best:.3f}")

    return history_y, np.array(chosen_points)


# ============================================
# 6. MAIN: RUN EVERYTHING + PLOT
# ============================================

def main():
    # 1) Generate synthetic dataset
    X, y = generate_dataset(n_samples=2000)
    print("Synthetic dataset generated.")
    print("X shape:", X.shape)
    print("y range: ", y.min(), "to", y.max())

    # 2) Train/test split (just to check GP performance if you want)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    # Train GP on training data
    gp, scaler = train_gp_model(X_train, y_train)

    # Optional: quick test performance check (not plotted)
    X_test_scaled = scaler.transform(X_test)
    y_pred, y_std = gp.predict(X_test_scaled, return_std=True)
    mse = np.mean((y_pred - y_test) ** 2)
    print(f"Test MSE of GP surrogate: {mse:.4f}")

    # 3) Run BO-style loop using the FIXED GP surrogate
    n_bo_iterations = 20
    history_y, chosen_points = bayesian_optimization_with_fixed_gp(
        gp, scaler, n_iterations=n_bo_iterations, n_candidates=1000
    )

    # 4) Plot yield vs BO iteration
    plt.figure(figsize=(7, 4))
    plt.plot(range(1, n_bo_iterations + 1), history_y, marker="o")
    plt.title("GP-based Optimization: Predicted Yield vs BO Iteration")
    plt.xlabel("BO Iteration")
    plt.ylabel("GP Predicted PHA Yield")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
