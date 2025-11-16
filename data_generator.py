import pandas as pd
import random
import numpy as np


class SampleDataGenerator():
    def __init__(self):
        #Defined some relevant parameters, and their ranges
        self.parameters = { "ph": [5, 7], "temp" : [20, 40], "carb_conc" : [2, 6], "carb_source" : ["glucose", "xylose", "oil"],
                            "oxygen" : [5, 15], "salinity" : [10, 30], "feeding_regime" : ["continuously_fed", "pulse_wise_feeding", "anaerobic_aerobic"]}

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

                    if len(rng) == 2:
                        value = random.uniform(rng[0], rng[1])
                    else:
                        index = random.randint(0, 2)
                        value = rng[index]
                    collection.append(value)

                container.append(collection)

            synt_data = pd.DataFrame(container, columns=head)
            return synt_data

    def depended_correlation(self, synt_data):
        #Trying to decide in which condition increase the yield,
        #Separated the generation of dependent variabile values because I needed the other parameters populated
        dependent_var = []
        for row in synt_data.itertuples():
            ph = row[0]
            temp = row[1]
            carb_conc = row[2]
            carb_source = row[3]
            oxygen = row[4]
            salinity = row[5]
            feeding_regime = [6]

            increase = False

            #From what I've seen from papers, usually the ph is between 7/7.5, temp usually between 25 and 30 degrees
            #and carbon concentration ranges around 2-3%
            #the other variables, I went with guesswork and some intuition
            if (ph >= 7 and ph < 8) and (temp >= 25 and temp < 30):
                if carb_conc >= 2 and carb_conc < 4:
                    if salinity < 15 and oxygen < 10:
                        increase = True
                elif carb_source == "glucose":
                    increase = True

            yield_value = 0
            if increase:
                yield_value = random.uniform(0.6, 1.2)
            else:
                yield_value = random.uniform(0.0, 0.5)

            dependent_var.append(yield_value)

        synt_data["yield"] = dependent_var
        return synt_data
