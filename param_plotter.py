import numpy as np
import matplotlib.pyplot as plt

# ==============================
# UPDATED RELATIONSHIP FUNCTIONS
# (matching new yield logic)
# ==============================

def quad_peak(x, x_opt, width):
    return np.maximum(0.0, 1.0 - ((x - x_opt) / width) ** 2)

def plateau_curve(x, half_sat):
    return x / (x + half_sat)

def decreasing_curve(x, half_sat):
    return (1 - x / (x + half_sat))

# ==============================
# UPDATED PARAMETER FUNCTIONS
# ==============================

params = {
    # pH: strong quadratic peak at 7.0
    "ph": {
        "range": np.linspace(6.5, 7.5, 300),
        "func": lambda x: 0.25 * quad_peak(x, 7.0, 0.25)
    },

    # Temperature: species ignored here, so average optimal at 32.5
    "temp": {
        "range": np.linspace(25, 40, 300),
        "func": lambda x: 0.20 * quad_peak(x, 32.5, 4.0)
    },

    # Stage 1 carbon: moderate peak
    "s1_carb_conc": {
        "range": np.linspace(10, 20, 300),
        "func": lambda x: 0.15 * quad_peak(x, 15.0, 4.0)
    },

    # Stage 1 agitation: wide gentle peak
    "s1_agitation": {
        "range": np.linspace(350, 850, 300),
        "func": lambda x: 0.15 * quad_peak(x, 600.0, 250.0)
    },

    # Stage 2 carbon conc: wide peak
    "s2_carb_conc": {
        "range": np.linspace(20, 60, 300),
        "func": lambda x: 0.20 * quad_peak(x, 40.0, 25.0)
    },

    # Stage 2 agitation
    "s2_agitation": {
        "range": np.linspace(300, 700, 300),
        "func": lambda x: 0.15 * quad_peak(x, 500.0, 250.0)
    },

    # Stage 1 nitrogen → plateau
    "s1_nitrogen": {
        "range": np.linspace(1, 3, 300),
        "func": lambda x: 0.10 * plateau_curve(x, half_sat=1.0)
    },

    # Stage 2 nitrogen → decreasing
    "s2_nitrogen": {
        "range": np.linspace(0.0, 0.03, 300),
        "func": lambda x: 0.25 * decreasing_curve(x, half_sat=0.01)
    },

    # Stage 1 time → plateau
    "s1_time": {
        "range": np.linspace(6, 20, 300),
        "func": lambda x: 0.15 * np.minimum(x / 14.0, 1.0)
    },

    # Stage 2 time → plateau
    "s2_time": {
        "range": np.linspace(10, 30, 300),
        "func": lambda x: 0.20 * np.minimum(x / 25.0, 1.0)
    }
}

# ==============================
# PLOT CONTINUOUS PARAMETERS
# ==============================

for param, info in params.items():
    x = info["range"]
    y = info["func"](x)

    plt.figure(figsize=(6,4))
    plt.plot(x, y)
    plt.title(f"Updated Relationship between {param} and PHA Content (%)")
    plt.xlabel(param)
    plt.ylabel("Relative Contribution to PHA Content")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ==============================
# CATEGORICAL PARAMETERS
# ==============================

categorical_params = {
    "species": ["Cupriavidus_necator", "Escherichia_coli"],
    "feeding_regime": ["continuously_fed", "pulse_wise_feeding"],
    "feedstock": ["sugars", "oils"]
}

# Updated categorical effects to match new scaling
cat_effects = {
    "species": [0.10, 0.05],
    "feeding_regime": [0.10, -0.02],
    "feedstock": [0.05, 0.10]
}

for param, cats in categorical_params.items():
    effects = cat_effects[param]

    plt.figure(figsize=(6,4))
    plt.bar(cats, effects)
    plt.title(f"Updated Categorical Relationship: {param} vs PHA Content (%)")
    plt.xlabel(param)
    plt.ylabel("Relative Contribution to PHA Content")
    plt.tight_layout()
    plt.show()