import numpy as np
from scipy.integrate import odeint
import pandas as pd

def main(seed):
    ### Define model
    seed_num = seed
    Nx = 40; # number of grid points along latitude circle
    F = 8;
    T = 1000
    def lorenz96(x, t): # right-hand side of the Lorenz 96 model
        n = len(x);
        dx = [(x[np.mod(i+1,n)]-x[i-2])*x[i-1]-x[i]+F for i in range(0,n)];
        # Q: Purpose of mod(i+1,n)?
        return dx
    #np.random.seed(42);  # for reproducibility
    rng_t = np.random.RandomState(seed_num); # for reproducibility
    x0 = rng_t.randn(Nx); # sample from 40 independent standard Gaussian RVs
    t = np.arange(0,1000,0.01); # time integration period
    x = odeint(lorenz96,x0,t); # [time, state dim]
    print("x.shape", x.shape)
    indices = rng_t.choice(50000, size=2000, replace=False)
    X_initial = x[indices, :]
    pd.DataFrame(X_initial).to_csv("X_initial_forecst_hristo_seed{}.csv".format(int(seed_num)))
    x_truth_trajectory = x[50000:50000+T*2:2, :]
    print("x_truth_trajectory.shape",x_truth_trajectory.shape)
    pd.DataFrame(x_truth_trajectory).to_csv("x_truth_trajectory_seed{}.csv".format(int(seed_num)))
    ## observation
    obs_sigma = 0.01
    y_observation_trajectory = np.arctan(x_truth_trajectory) + obs_sigma * rng_t.randn(T, Nx)
    print("y_observation_trajectory.shape",y_observation_trajectory.shape)
    pd.DataFrame(y_observation_trajectory).to_csv("y_observation_trajectory_seed{}.csv".format(int(seed_num)))
    print("X_initial.shape",X_initial.shape)
    print("initial rmse=",np.sqrt(((X_initial.mean(axis=0) - x_truth_trajectory[0,:])**2).mean()))
    print("\n")


for i in np.arange(1):
    main(i)




