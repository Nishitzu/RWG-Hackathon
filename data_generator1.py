import pandas as pd
import random
import numpy as np


class SampleDataGenerator():
    def __init__(self):
        #Defined some relevant parameters, and their ranges
        # Expanded to match full PHA bioprocess conditions
        self.parameters = { 
            "species": ["Cupriavidus_necator", "Escherichia_coli"],

            # General
            "ph": [6.8, 7.2],
            "temp": [27, 37],

            # Stage 1: Growth Phase
            "s1_carb_conc": [10, 20],
            "s1_nitrogen": [1, 3],
            "s1_agitation": [400, 800],
            "s1_time": [6, 20],

            # Stage 2: Induction Phase
            "s2_carb_conc": [20, 60],
            "s2_nitrogen": [0.00, 0.02],
            "s2_agitation": [300, 700],
            "s2_time": [10, 30],

            # Extra categorical parameters
            "feeding_regime": ["continuously_fed", "pulse_wise_feeding"],
            "feedstock": ["sugars", "oils"]
        }

    def syntetic_data_gen(self):
            #list container for the all the sampled lists, it will be the structure which builds the dataframe
            container = []
            #header with column names
            head = []
            #an arbitrary number of samples
            num_of_samples = 3000

            #Iterating over each param in the dictionary to create some data
            #I am literally guessing at this point, if I have to be honest
            for i in range(0, num_of_samples+1):
                collection = []
                for param, rng in self.parameters.items():
                    print(param, range)
                    if i == 0:
                        head.append(param)

                    # numeric parameters
                    if len(rng) == 2 and not isinstance(rng[0], str):
                        value = random.uniform(rng[0], rng[1])
                    # categorical parameters
                    else:
                        index = random.randint(0, len(rng)-1)
                        value = rng[index]

                    collection.append(value)

                container.append(collection)

            synt_data = pd.DataFrame(container, columns=head)
            return synt_data

    def depended_correlation(self, synt_data):
        #Trying to decide in which condition increase the yield,
        #Separated the generation of dependent variabile values because I needed the other parameters populated
        dependent_var = []

        # helper functions
        def quad_peak(x, x_opt, width):
            # returns 0 → 1
            val = 1.0 - ((x - x_opt) / width) ** 2
            return max(0.0, val)

        def plateau(x, half_sat):
            return x / (x + half_sat)

        def decreasing(x, half_sat):
            return (1 - x / (x + half_sat))


        for row in synt_data.itertuples():

            species        = row.species
            ph             = row.ph
            temp           = row.temp

            s1_carb        = row.s1_carb_conc
            s1_nitrogen    = row.s1_nitrogen
            s1_agitation   = row.s1_agitation
            s1_time        = row.s1_time

            s2_carb        = row.s2_carb_conc
            s2_nitrogen    = row.s2_nitrogen
            s2_agitation   = row.s2_agitation
            s2_time        = row.s2_time

            feeding_regime = row.feeding_regime
            feedstock      = row.feedstock


            # -------------------------
            #          LOGIC
            # -------------------------

            # Start with a neutral baseline
            score = 0.0

            # Species effect
            score += 0.10 if species == "Cupriavidus_necator" else 0.05

            # QUADRATIC: pH
            score += 0.25 * quad_peak(ph, 7.0, 0.25)

            # QUADRATIC: temperature (different optima)
            if species == "Cupriavidus_necator":
                score += 0.20 * quad_peak(temp, 30.0, 4.0)
            else:
                score += 0.20 * quad_peak(temp, 35.0, 4.0)

            # QUADRATIC: Stage 1 carbon
            score += 0.15 * quad_peak(s1_carb, 15.0, 4.0)

            # Stage 1 nitrogen → plateau
            score += 0.10 * plateau(s1_nitrogen, 1.0)

            # QUADRATIC: Stage 1 agitation
            score += 0.15 * quad_peak(s1_agitation, 600.0, 250.0)

            # Stage 1 time plateau
            score += 0.15 * min(s1_time / 14.0, 1.0)

            # QUADRATIC: Stage 2 carbon
            score += 0.20 * quad_peak(s2_carb, 40.0, 25.0)

            # Stage 2 nitrogen → decreasing
            score += 0.25 * decreasing(s2_nitrogen, 0.01)

            # QUADRATIC: Stage 2 agitation
            score += 0.15 * quad_peak(s2_agitation, 500.0, 250.0)

            # Stage 2 time plateau
            score += 0.20 * min(s2_time / 25.0, 1.0)

            # Feedstock
            score += 0.10 if feedstock == "oils" else 0.05

            # Feeding regime
            score += 0.10 if feeding_regime == "continuously_fed" else -0.02

            # Add noise
            score += random.uniform(-0.05, 0.05)

            # -------------------------
            # SCALE to 0.2 → 2.0 range
            # -------------------------

            # Maximum possible (approx)
            max_score = 2.6

            # Normalize to 0 → 1
            normalized = max(0, min(1, score / max_score))

            # Final yield
            yield_val = 0.2 + normalized * (2.0 - 0.2)

            dependent_var.append(yield_val)

        synt_data["yield"] = dependent_var
        return synt_data


#ph, temp, carbon conc, carbon src, nitrogen