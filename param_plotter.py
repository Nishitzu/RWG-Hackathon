import numpy as np
import matplotlib.pyplot as plt

# Quadratic peak helper
def quad_peak(x, x_opt, width, scale=1.0):
    val = 1.0 - ((x - x_opt) / width) ** 2
    return scale * np.maximum(0.0, val)

# Logistic / plateau helper
def plateau_curve(x, half_sat, scale=1.0):
    return scale * (x / (x + half_sat))

# Monotonic decreasing (inverse logistic)
def decreasing_curve(x, half_sat, scale=1.0):
    return scale * (1 - x / (x + half_sat))


# ---------------------------------------------------------
# CONTINUOUS PARAMETER DEFINITIONS
# ---------------------------------------------------------

params = {
    "ph": {
        "range": np.linspace(6.0, 8.0, 200),
        "func": lambda x: quad_peak(x, 7.0, 0.4, 1.0)
    },
    "temperature": {
        "range": np.linspace(25, 40, 200),
        "func": lambda x: quad_peak(x, 30.0, 5.0, 1.0)
    },
    "s1_carb_conc": {
        "range": np.linspace(10, 20, 200),
        "func": lambda x: quad_peak(x, 15.0, 5.0, 0.8)
    },
    "s1_agitation": {
        "range": np.linspace(400, 800, 200),
        "func": lambda x: quad_peak(x, 600.0, 200.0, 0.8)
    },
    "s2_carb_conc": {
        "range": np.linspace(20, 60, 200),
        "func": lambda x: quad_peak(x, 40.0, 20.0, 0.8)
    },
    "s2_agitation": {
        "range": np.linspace(300, 700, 200),
        "func": lambda x: quad_peak(x, 500.0, 200.0, 0.6)
    },

    # -------------------------
    # NEW NITROGEN PARAMETERS
    # -------------------------

    "s1_nitrogen": {
        "range": np.linspace(1, 3, 200),
        "func": lambda x: plateau_curve(x, half_sat=1.0, scale=0.7)
    },

    "s2_nitrogen": {
        "range": np.linspace(0.0, 0.05, 200),
        "func": lambda x: decreasing_curve(x, half_sat=0.01, scale=1.0)
    },
}


# ---------------------------------------------------------
# CONTINUOUS PARAMETER PLOTS
# ---------------------------------------------------------

for param, info in params.items():
    x = info["range"]
    y = info["func"](x)

    plt.figure(figsize=(6,4))
    plt.plot(x, y)
    plt.title(f"Relationship between {param} and PHA Content (%)")
    plt.xlabel(param)
    plt.ylabel("Relative PHA Content (%)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------
# CATEGORICAL PARAMETERS
# ---------------------------------------------------------

categorical_params = {
    "species": ["Cupriavidus_necator", "Escherichia_coli"],
    "feeding_regime": ["continuously_fed", "pulse_wise_feeding"],
    "feedstock": ["sugars", "oils"]
}

cat_effects = {
    "species": [1.0, 0.6],          # Cn > Ec
    "feeding_regime": [0.8, 0.5],   # continuous > pulse
    "feedstock": [0.7, 0.9]         # oils > sugars
}

for param, cats in categorical_params.items():
    effects = cat_effects[param]
    plt.figure(figsize=(6,4))
    plt.bar(cats, effects)
    plt.title(f"Categorical Relationship: {param} vs PHA Content (%)")
    plt.xlabel(param)
    plt.ylabel("Relative PHA Content (%)")
    plt.tight_layout()
    plt.show()
