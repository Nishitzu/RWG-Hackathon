The UI Mockup that was shown in the 5-minute presentation is linked here: https://www.figma.com/proto/aJX91G0JPyl9rZpJuHWgtF/brishti.miller22-s-team-library?node-id=3343-920&t=8vYbj6upNN8oXYBj-1&starting-point-node-id=3343%3A920

We created a Bayesian Optimiser software called PHAntastic. PHAntastic allows a researcher to screen millions of bioreactor parameter combinations, in just a couple hundred experiments or less! This drastically reduces R&D costs, labour, and time.

Here is how PHAntastic works:
The PHAntastic Bayesian optimiser starts with zero or just a small handful of initial random experiments.
It builds its own model internally (usually a Gaussian Process).
It updates the model ongoingly as new experiments are run.
It proposes the next best experiment each cycle.

We didn't have access to a wet lab, so we decided to validate PHAntastic with synthetic bioreactor data. To get this bioreactor data, we made a virtual bioreactor simulation using a neural network machine learning algorithm. This involved training the neural network to act as a surrogate model. Check out sample_synthetic_data.xlsx to see what this synthetic data looked like! 

Now that we have lots of data, we started to validate the feasibility of PHAntastic in recommending experiments to improve PHA titre. We gave the PHAntastic Bayesian Optimiser only 3 synthetic data points, and then it gave us what it thought the next best parameter settings were. As you can see in the following graph, PHAntastic successfully recommended the next best experiments to get optimal PHA titre! It took only 20 iterations (so only 60 experiments in total!) to get to a high titre.

<img width="1198" height="728" alt="image" src="https://github.com/user-attachments/assets/cfd60ad3-046b-4799-94cb-2304ce9795f4" />


All the code for producing synthetic data, training the NN, and for the PHAntastic Bayesian Optimiser is found in NNModel_Bayesian_Optimizer_main.py. 

Here is the link to our Powerpoint Presentation:
https://onedrive.live.com/:p:/g/personal/3FFDCFC027C68A88/EY37whq9coBHmbqevmGcyEIBsUOFbmHwtGT7f_qeWy3I-g?resid=3FFDCFC027C68A88!s1ac2fb8d72bd478099ba9ebe619cc842&ithint=file%2Cpptx&e=YLmNfV&migratedtospo=true&redeem=aHR0cHM6Ly8xZHJ2Lm1zL3AvYy8zZmZkY2ZjMDI3YzY4YTg4L0VZMzd3aHE5Y29CSG1icWV2bUdjeUVJQnNVT0ZibUh3dEdUN2ZfcWVXeTNJLWc_ZT1ZTG1OZlY&previoussessionid=7941bad5-8eb5-b9f2-b021-b6bca436f318
