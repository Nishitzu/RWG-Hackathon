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
        def quad_peak(x, x_opt, width, scale=1.0):
            val = 1.0 - ((x - x_opt) / width) ** 2
            return scale * max(0.0, val)

        def plateau(x, half_sat, scale=1.0):
            return scale * (x / (x + half_sat))

        def decreasing(x, half_sat, scale=1.0):
            return scale * (1 - x / (x + half_sat))

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

            yield_val = 0.0

            # Species effect
            if species == "Cupriavidus_necator":
                yield_val += 0.15
            else:
                yield_val += 0.05

            # QUADRATIC: pH
            yield_val += quad_peak(ph, 7.0, 0.4, 1.0)

            # QUADRATIC: temperature
            if species == "Cupriavidus_necator":
                yield_val += quad_peak(temp, 30.0, 5.0, 1.0)
            else:
                yield_val += quad_peak(temp, 35.0, 5.0, 1.0)

            # QUADRATIC: Stage 1 carbon
            yield_val += quad_peak(s1_carb, 15.0, 5.0, 0.8)

            # NEW: Stage 1 nitrogen → PLATEAU (more nitrogen → more growth)
            yield_val += plateau(s1_nitrogen, half_sat=1.0, scale=0.7)

            # QUADRATIC: Stage 1 agitation
            yield_val += quad_peak(s1_agitation, 600.0, 200.0, 0.8)

            # Stage 1 time → plateau around 12 h
            yield_val += min(s1_time / 12.0, 1.0)

            # QUADRATIC: Stage 2 carbon
            yield_val += quad_peak(s2_carb, 40.0, 20.0, 0.8)

            # NEW: Stage 2 nitrogen → strictly decreasing (low N → high PHA)
            yield_val += decreasing(s2_nitrogen, half_sat=0.01, scale=1.2)

            # QUADRATIC: Stage 2 agitation
            yield_val += quad_peak(s2_agitation, 500.0, 200.0, 0.6)

            # Stage 2 time → plateau around 20 h
            yield_val += min(s2_time / 20.0, 1.0)

            # Feedstock
            if feedstock == "oils":
                yield_val += 0.35
            else:
                yield_val += 0.20

            # Feeding regime
            if feeding_regime == "continuously_fed":
                yield_val += 0.15
            else:
                yield_val -= 0.05

            # Noise
            yield_val += random.uniform(-0.2, 0.2)

            # Clamp to realistic bounds
            yield_val = max(0.2, min(2.0, yield_val))

            dependent_var.append(yield_val)

        synt_data["yield"] = dependent_var
        return synt_data
