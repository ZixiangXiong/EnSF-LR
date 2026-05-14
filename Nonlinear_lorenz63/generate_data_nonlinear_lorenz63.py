import numpy as np
from scipy.integrate import odeint
import pandas as pd

def main(seed):
    ### Define model
    seed_num = seed
    Nx = 3
    T = 300
    sigma = 10.0
    rho = 28.0
    beta = 8.0 / 3.0
    def lorenz63(state, t): # right-hand side of the Lorenz 96 model
        x, y, z = state
        dx_dt = sigma * (y - x)
        dy_dt = x * (rho - z) - y
        dz_dt = x * y - beta * z
        return [dx_dt, dy_dt, dz_dt]

    rng_t = np.random.RandomState(seed_num); # for reproducibility
    x0 = rng_t.randn(Nx);
    t = np.arange(0,1000,0.01); # time integration period
    x = odeint(lorenz63,x0,t); # [time, state dim]
    print("x.shape", x.shape)
    indices = rng_t.choice(50000, size=1000, replace=False)
    X_initial = x[indices, :]
    pd.DataFrame(X_initial).to_csv("X_initial_forecst_hristo_lorenz63_seed{}.csv".format(int(seed_num)))
    x_truth_trajectory = x[50000:50000+T*5:5, :]
    print("x_truth_trajectory_lorenz63.shape",x_truth_trajectory.shape)
    pd.DataFrame(x_truth_trajectory).to_csv("x_truth_trajectory_lorenz63_seed{}.csv".format(int(seed_num)))
    ## observation
    obs_sigma = 0.001
    y_observation_trajectory = np.arctan(x_truth_trajectory) + obs_sigma * rng_t.randn(T,Nx)
    print("y_observation_trajectory.shape",y_observation_trajectory.shape)
    pd.DataFrame(y_observation_trajectory).to_csv("y_observation_trajectory_lorenz63_seed{}.csv".format(int(seed_num)))
    print("X_initial.shape",X_initial.shape)
    print("initial rmse=",np.sqrt(((X_initial.mean(axis=0) - x_truth_trajectory[0,:])**2).mean()))
    print("\n")


for i in np.arange(20):
    main(i)