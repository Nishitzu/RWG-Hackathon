import matplotlib.pyplot as plt
import pandas as pd
from data_generator1 import SampleDataGenerator
from gp_and_aq_func import run_bayesian_optimization


############################################################
# FILE: bo_run_and_plot.py
# PURPOSE:
#   - Generate initial dataset
#   - Reduce to selected parameters ONLY
#   - Run Bayesian Optimization (GP surrogate stays FIXED)
#   - Plot BO iteration vs best yield
############################################################


def run_bo_and_plot(iterations=20, lenscale=1.5):
    # 1. Generate starting synthetic dataset
    gen = SampleDataGenerator()
    df = gen.syntetic_data_gen()
    df = gen.depended_correlation(df)

    # --------------------------------------------------------
    # LIMIT DATASET TO YOUR 4 PARAMETERS + YIELD
    # --------------------------------------------------------
    selected_params = ["ph", "temp", "s2_carb_conc", "s2_agitation", "yield"]
    df = df[selected_params].copy()

    print("Reduced dataset shape:", df.shape)
    print("Initial best yield:", df["yield"].max())

    # 2. Run BO loop
    history, X_final, y_final = run_bayesian_optimization(
        initial_data=df,
        iterations=iterations,
        lenscale=lenscale
    )

    print("Final best yield after BO:", max(history))

    # 3. Plot BO iteration vs best yield
    plt.figure(figsize=(7, 4))
    plt.plot(history, marker="o")
    plt.title("Bayesian Optimization – Best Yield vs Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Best Yield")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return history, X_final, y_final


if __name__ == "__main__":
    run_bo_and_plot()
