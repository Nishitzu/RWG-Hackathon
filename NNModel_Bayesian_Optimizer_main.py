import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

from tqdm import tqdm


# ============================================================
# 1. PARAMETER SPACE + TRUE UNDERLYING TITRE FUNCTION
# ============================================================

PARAM_RANGES = {
    "ph": (6.8, 7.2),
    "temp": (27.0, 37.0),
    "s2_carb_conc": (20.0, 60.0),
    "s2_agitation": (300.0, 700.0),
}


def quad_peak(x, x_opt, width):
    """Simple quadratic peak in [0, 1]."""
    val = 1.0 - ((x - x_opt) / width) ** 2
    return np.maximum(0.0, val)


def true_titre_function(ph, temp, s2_carb, s2_agit):
    """
    TRUE PHA TITRE model (20–110 g/L)
    Used both for:
    - generating the training data
    - evaluating TRUE titre during BO
    """

    score_ph   = quad_peak(ph,       7.0, 0.2)
    score_temp = quad_peak(temp,    32.0, 5.0)
    score_carb = quad_peak(s2_carb, 40.0, 15.0)
    score_agit = quad_peak(s2_agit, 500.0, 150.0)

    score = (
        0.30 * score_ph +
        0.25 * score_temp +
        0.25 * score_carb +
        0.20 * score_agit
    )

    # Noise to simulate experiment variation
    noise = np.random.normal(0, 0.015, size=np.shape(score))
    score_noisy = np.clip(score + noise, 0.0, 1.0)

    # Map normalized score [0, 1] → [20, 110] g/L TITRE
    titre = 20 + score_noisy * 90
    return titre


# ============================================================
# 2. SYNTHETIC DATA (4000 samples)
# ============================================================

def sample_uniform_params(n):
    ph   = np.random.uniform(*PARAM_RANGES["ph"], size=n)
    temp = np.random.uniform(*PARAM_RANGES["temp"], size=n)
    carb = np.random.uniform(*PARAM_RANGES["s2_carb_conc"], size=n)
    agit = np.random.uniform(*PARAM_RANGES["s2_agitation"], size=n)
    return np.vstack([ph, temp, carb, agit]).T


def generate_dataset(n_samples=4000):
    X = sample_uniform_params(n_samples)
    y = true_titre_function(X[:, 0], X[:, 1], X[:, 2], X[:, 3])
    return X, y


# ============================================================
# 3. TRAIN NEURAL NETWORK MODEL (WITH EPOCHS + LOSS)
# ============================================================

def train_nn_model(X, y, n_epochs=5):
    """
    Train an MLPRegressor as a surrogate model.
    Simulated epochs using warm_start=True and max_iter=1.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    mlp = MLPRegressor(
        hidden_layer_sizes=(64, 64),
        activation='relu',
        solver='adam',
        learning_rate_init=1e-3,
        max_iter=1,
        warm_start=True,
        random_state=0
    )

    losses = []

    print("\nTraining Neural Network surrogate...\n")
    for epoch in tqdm(range(n_epochs), desc="NN Training Epochs"):
        mlp.fit(X_scaled, y)
        y_pred = mlp.predict(X_scaled)
        mse = np.mean((y_pred - y) ** 2)
        losses.append(mse)

    print(f"\nFinal training MSE: {losses[-1]:.6f}")
    return mlp, scaler, losses


# ============================================================
# 4. BO-STYLE LOOP USING FIXED NN SURROGATE
# ============================================================

def bo_with_fixed_model(model, scaler, n_iterations=20, n_candidates=3):
    """
    NN-based optimization loop over 4D parameter space.
    Tracks BEST TRUE TITRE at each iteration.
    """
    best_true_y = -np.inf
    history_best_true = []
    chosen_points = []

    print("\nRunning model-based optimization (NN surrogate)...\n")

    for it in tqdm(range(n_iterations), desc="BO Iterations"):
        X_cand = sample_uniform_params(n_candidates)
        X_cand_scaled = scaler.transform(X_cand)

        y_pred = model.predict(X_cand_scaled)

        best_idx = np.argmax(y_pred)
        x_best = X_cand[best_idx]

        true_y = true_titre_function(*x_best)

        best_true_y = max(best_true_y, true_y)
        history_best_true.append(best_true_y)
        chosen_points.append(x_best)

    return history_best_true, np.array(chosen_points)


# ============================================================
# 5. MAIN: RUN EVERYTHING + PLOTS
# ============================================================

def main():
    # CHANGE 1: Train on 4000 samples
    X, y = generate_dataset(10000)
    print("\nDataset generated:", X.shape, "inputs, y shape:", y.shape)
    print("Titre range:", float(y.min()), "to", float(y.max()))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    n_epochs = 15
    model, scaler, losses = train_nn_model(X_train, y_train, n_epochs=n_epochs)

    X_test_scaled = scaler.transform(X_test)
    y_test_pred = model.predict(X_test_scaled)
    test_mse = np.mean((y_test_pred - y_test) ** 2)
    print(f"Test MSE: {test_mse:.6f}")

    # CHANGE 2: Give BO only 2–3 initial candidates
    n_bo_iterations = 20
    history_best_true, chosen_points = bo_with_fixed_model(
        model, scaler,
        n_iterations=n_bo_iterations,
        n_candidates=3        # <--- only 3 initial examples each iteration
    )

    import matplotlib as mpl

    # =============================
    # Use Aptos Display for ALL text
    # =============================
    mpl.rcParams['font.family'] = 'DejaVu Sans'
    mpl.rcParams['font.sans-serif'] = ['DejaVu Sans']

    # ============================
    # Plot 1: Epoch vs training loss (Aptos Display)
    # ============================
    plt.figure(figsize=(8, 5), dpi=300)

    plt.plot(
        range(1, n_epochs + 1),
        losses,
        marker='o',
        markersize=6,
        linewidth=2.5,
        color="#1f77b4"
    )

    plt.title("Neural Network Training Curve", fontsize=16, fontweight="bold")
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Training Loss (MSE)", fontsize=14)

    plt.grid(True, linestyle="--", alpha=0.4)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.tight_layout()
    plt.savefig("NN_TRAINING_LOSS_EPOCHS.png", dpi=300)
    plt.show()
    plt.close()


    # ============================
    # Plot 2: BO iteration vs best TRUE TITRE (Aptos Display)
    # ============================
    plt.figure(figsize=(8, 5), dpi=300)

    plt.plot(
        range(1, n_bo_iterations + 1),
        history_best_true,
        marker='o',
        markersize=6,
        linewidth=2.5,
        color="#d62728"
    )

    plt.title("Bayesian Optimisation Progress: Best Titre", fontsize=16, fontweight="bold")
    plt.xlabel("BO Iteration", fontsize=14)
    plt.ylabel("Best Titre (g/L)", fontsize=14)

    # SMART Y-LIMITS
    y_min = min(history_best_true)
    y_max = max(history_best_true)
    margin = (y_max - y_min) * 0.15
    if margin == 0:
        margin = 2
    plt.ylim(y_min - margin, y_max + margin)

    plt.grid(True, linestyle="--", alpha=0.4)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xticks(np.arange(0, n_bo_iterations + 1, 5))
    plt.yticks(np.arange(int(y_min - margin), int(y_max + margin), 5))
    plt.tight_layout()
    plt.savefig("BO_ITERATION_VS_TITRE.png", dpi=300)
    plt.show()
    plt.close()


    print(history_best_true)


if __name__ == "__main__":
    main()