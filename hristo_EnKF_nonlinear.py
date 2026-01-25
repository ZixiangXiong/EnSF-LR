import numpy as np
from scipy.integrate import odeint
import pandas as pd
#import seaborn as sns

### Define model
Nx = 40; # number of grid points along latitude circle
F = 8;
read_file = 1
### Ensemble size
Ne_total = 2000;
def lorenz96(x, t): # right-hand side of the Lorenz 96 model
    n = len(x);
    dx = [(x[np.mod(i+1,n)]-x[i-2])*x[i-1]-x[i]+F for i in range(0,n)];
    # Q: Purpose of mod(i+1,n)?
    return dx

if read_file == 0:
    np.random.seed(seed=10); # for reproducibility
    x0 = np.random.randn(Nx); # sample from 40 independent standard Gaussian RVs
    t = np.arange(0,30,0.01); # time integration period
    x = odeint(lorenz96,x0,t); # [time, state dim]
    xt = x[-1,:]; # last state
    xp = xt+0.05*np.random.randn(Nx);
    ### Initial ensemble
    ones = np.ones(Ne_total);
    Xf = np.outer(ones,xp) + 0.05*np.random.randn(Ne_total,Nx); # defined from perturbed run
    # np.outer(ones,xb) repeats xp on each row
    print('Xf shape', Xf.shape) # [Ne, Nx]

    ### Propagate ensemble
    time = np.arange(0,10,0.01); # shorter than before
    M = len(time);
    snapshots = np.zeros((Ne_total,M,Nx)); # [Ne, Nt, Nx]
    for e in range(0,Ne_total): # loop over ensemble and propagate each member
        sol = odeint(lorenz96,Xf[e,:],time); # [Nt, Nx]
        #print(sol.shape)
        snapshots[e,:,:] = sol;
        Xf[e,:] = sol[-1,:]; # sol[-1,:] = last time only, all state vars

    ### Propagate true state
    xSt = odeint(lorenz96,xt,time);
    xt = xSt[-1,:]; # last time only, all state vars
    # Save forecast ensemble (2000 × 40)
    np.savetxt("X_initial_forecst_hristo.csv", Xf, delimiter=",")
    # Save truth state (40,)
    np.savetxt("x_initial_truth_hristo.csv", xt, delimiter=",")
elif read_file == 1:
    # Load initial forecast ensemble(2000 × 40)
    Xf = np.loadtxt("X_initial_forecst_hristo.csv", delimiter=",")
    # Load initial truth state(40,)
    xt = np.loadtxt("x_initial_truth_hristo.csv", delimiter=",")





### Declarations
np.random.seed(seed=42);  # to ensure results are reproducible
time_step = 0.1;  # dt for numerical integration
p = 0.25;  # fraction of the directly observed state variables
obs_type="fix_even"
linear_regression = 1
T = 1000;  # number of time steps to simulate (not the absolute time)
errorf_k = np.zeros(T);  # array to store the forecast errors
errora_k = []  # array to store the analysis errors
errora_obs = []
errora_unobs = []
I = np.eye(Nx, Nx);  # identity matrix
Ne = 2000;  # the ensemble size we will use in this experiment
obs_gap = 1
ones = np.ones(Ne);  # a vector of 1s, to be used later in creating
# the full ensemble matrices
loc = 3;  # localization radius to be used in this experiment
infl = 1.02;  # inflation factor to be used in this experiment
### Ensemble size

### Initial true state and ensemble
xt_k = xt.copy();  # this is the last true state from the
# previous time integration
ind = np.random.permutation(np.arange(0, Ne_total))[:Ne];
Xf_k = Xf[ind,:].copy().T;
Xa_k = np.zeros_like(Xf_k)
errora_k.append(np.sqrt(((Xf_k.mean(axis=1) - xt_k) ** 2).mean()))


### Observation-related variables
Ny = int(round(p * Nx));  # number of observations
sig_obs = 0.005;  # ob error std
R_k = (sig_obs ** 2) * np.eye(Ny, Ny);  # ob error covariance matric
if obs_type == "fix_even":
    # Observation at time level k
    obs_comp = np.arange(0, Nx, int(1/p))[0:Ny]
    unobs_comp = np.sort(np.setdiff1d(np.arange(Nx), obs_comp))


### Loop over time and assimilate observations (when present)
for k in range(0, T):
    print("\n")
    print(f'Assimilation cycle: {k}');

    # Forecast mean
    xf_k = np.mean(Xf_k, 1);

    # (Localized) forecast covariance
    #L = create_localization_matrix(loc, Nx);
    #Pf_k = np.cov(Xf_k);

    # RMSE (L2 or Euclidean norm) of forecast mean
    #errorf_k[k] = np.linalg.norm(xf_k - xt_k);
    errorf_k[k] = np.sqrt(((xf_k - xt_k) ** 2).mean())
    print("forecast error=",errorf_k[k])
    print("obs error forecast=", np.sqrt(((Xf_k.mean(axis=1)[obs_comp] - xt_k[obs_comp]) ** 2).mean()))
    print("unobs error forecast=",np.sqrt(((Xf_k.mean(axis=1)[unobs_comp] - xt_k[unobs_comp]) ** 2).mean()))

    if k % obs_gap == 0:
        print("DA will be done")
        if obs_type == "random":
            # Observation at time level k
            obs_comp = np.random.permutation(Nx);
            obs_comp = np.sort(obs_comp[0:Ny]);  # randomly select Ny state components
            # (different each cycle)

        if linear_regression == 0:
            H_k = I[obs_comp, :];  # ob operator
            err_obs_k = sig_obs * np.random.randn(Ny);
            y_k = np.arctan(H_k @ xt_k) + err_obs_k;  # true observation value
            # Perturbed observations - one for each forecast member
            Eobs_k = sig_obs * np.random.randn(Ny, Ne);
            Yobs_k = np.outer(y_k, np.ones(Ne)) + Eobs_k;  # [Ny,Ne]

            # Scaled innovation matrix
            Yf_k = np.arctan(H_k @ Xf_k)    # [Ny,Ne]
            Xf_k_center = (Xf_k - np.mean(Xf_k, axis=1, keepdims=True)) / np.sqrt(Ne - 1)   # [Nx,Ne]
            Yf_k_center = (Yf_k - np.mean(Yf_k, axis=1, keepdims=True)) / np.sqrt(Ne - 1)   # [Ny,Ne]
            D_k = Yobs_k - Yf_k;  # [Ny,Ne]
            #IN_k = R_k + H_k @ Pf_k @ H_k.T;  # [Ny,Ny]
            IN_k = R_k + Yf_k_center @ Yf_k_center.T;  # [Ny,Ny]
            Z_k = np.linalg.solve(IN_k, D_k);  # this gives (IN)^-1 * D =: Z [Ny,Ne]

            # EnKF update
            #Xa_k = Xf_k + Pf_k @ H_k.T @ Z_k;  # [Nx,Ne]
            Xa_k = Xf_k + Xf_k_center @ Yf_k_center.T @ Z_k;  # [Nx,Ne]
        elif linear_regression == 1 and Ny < Nx:
            H_k = I[obs_comp, :];  # ob operator
            err_obs_k = sig_obs * np.random.randn(Ny);
            y_k = np.arctan(H_k @ xt_k) + err_obs_k;  # true observation value
            # Perturbed observations - one for each forecast member
            Eobs_k = sig_obs * np.random.randn(Ny, Ne);
            Yobs_k = np.outer(y_k, np.ones(Ne)) + Eobs_k;  # [Ny,Ne]
            # Scaled innovation matrix
            Yf_k = np.arctan(H_k @ Xf_k)  # [Ny,Ne]
            Xf_k_center = (H_k @ Xf_k - np.mean(H_k @ Xf_k, axis=1, keepdims=True)) / np.sqrt(Ne - 1)  # [Ny,Ne]
            Yf_k_center = (Yf_k - np.mean(Yf_k, axis=1, keepdims=True)) / np.sqrt(Ne - 1)  # [Ny,Ne]
            D_k = Yobs_k - Yf_k;  # [Ny,Ne]
            IN_k = R_k + Yf_k_center @ Yf_k_center.T;  # [Ny,Ny]
            Z_k = np.linalg.solve(IN_k, D_k);  # this gives (IN)^-1 * D =: Z [Ny,Ne]
            """
            if k==600:
                pd.DataFrame(Xf_k[[0,4],:].T,columns=["x0", "x4"]).to_csv("data_test_20gap_step600_forecast.csv",index=False)
                pd.DataFrame(xt_k[[0,4]]).to_csv("data_test_20gap_step600_truth.csv",index=False)
                pd.DataFrame(y_k[[0, 1]]).to_csv("data_test_20gap_step600_obs.csv", index=False)
            """
            # EnKF update
            Xa_k[obs_comp, :] = Xf_k[obs_comp, :] + Xf_k_center @ Yf_k_center.T @ Z_k
            print("obs_analysis error pre=", np.sqrt(((Xa_k.mean(axis=1)[obs_comp] - xt_k[obs_comp]) ** 2).mean()))
            print("unobs_analysis error pre=",np.sqrt(((Xf_k.mean(axis=1)[unobs_comp] - xt_k[unobs_comp]) ** 2).mean()))

            # Linear regression update
            print("LR will be done")
            z_prior = np.arctan(H_k @ Xf_k)  ## shape (Ny,Ne)
            z_prior_mean = z_prior.mean(axis=1).reshape(Ny, 1)  ## shape (Ny,1)
            cov_zz = (z_prior - z_prior_mean) @ (z_prior - z_prior_mean).T / (Ne - 1)  ### shape (Ny,Ny)
            if np.linalg.matrix_rank(cov_zz) == cov_zz.shape[0]:
                inv_cov_zz = np.linalg.inv(cov_zz)
            else:
                inv_cov_zz = np.linalg.pinv(cov_zz)  # fallback
            x_prior_unobs_mean = Xf_k[unobs_comp, :].mean(axis=1).reshape(Nx - Ny, 1)  ### shape (Nx-Ny,1)
            cov_xz = (((Xf_k[unobs_comp, :] - x_prior_unobs_mean) @ (z_prior - z_prior_mean).T) / (Ne - 1)).reshape(Nx - Ny, Ny)  # shape (Nx-Ny,Ny)
            ## update unobserved states of Xa_k:
            #dx = (cov_xz @ inv_cov_zz) @ dz  ### shape (Nx-Ny,Ne)
            dx = cov_xz @ np.linalg.solve(cov_zz + R_k, D_k)
            Xa_k[unobs_comp, :] = Xf_k[unobs_comp, :] + dx  ### shape (Nx-Ny,Ne)
    else:
        Xa_k = Xf_k.copy()


    # RMSE of analysis mean
    """if k == 600:
        pd.DataFrame(Xa_k[[0,4], :].T, columns=["x0", "x4"]).to_csv("data_test_10gap_step600_analysis.csv",index=False)
        break"""
    xa_k = np.mean(Xa_k, 1)
    #errora_k[k] = np.linalg.norm(xa_k - xt_k);
    errora_k.append(np.sqrt(((xa_k - xt_k) ** 2).mean()))
    errora_obs.append(np.sqrt(((xa_k[obs_comp] - xt_k[obs_comp]) ** 2).mean()))
    errora_unobs.append(np.sqrt(((xa_k[unobs_comp] - xt_k[unobs_comp]) ** 2).mean()))
    print("analysis error=", errora_k[-1])
    print("obs_analysis error=", errora_obs[-1])
    print("unobs_analysis error=", errora_unobs[-1])

    # Forecast to next time (both ensemble and nature run)
    for e in range(0, Ne):
        Xa_e_S = odeint(lorenz96, Xa_k[:, e], [0, time_step]);
        #Xf_k[:, e] = Xa_e_S[-1, :]
        Xf_k[:, e] = Xa_e_S[-1, :] + np.random.normal(0.0, 0.5, Nx);
    xt_S = odeint(lorenz96, xt_k, [0, time_step]);
    xt_k = xt_S[-1, :];

df = pd.DataFrame(errora_k, columns=["RMSE_enkf"])
if linear_regression == 0:
    df.to_csv("RMSE_hristo_EnKF_nonlinear_{}%obs_{}obsgap.csv".format(int(p*100.), obs_gap))
    if p < 1:
        pd.DataFrame(errora_obs, columns=["RMSE_obs"]).to_csv("RMSE_hristo_EnKF_nonlinear_observed_{}%obs_{}obsgap.csv".format(int(p * 100.), obs_gap))
        pd.DataFrame(errora_unobs, columns=["RMSE_unobs"]).to_csv("RMSE_hristo_EnKF_nonlinear_unobserved_{}%obs_{}obsgap.csv".format(int(p * 100.), obs_gap))
elif linear_regression == 1:
    df.to_csv("RMSE_hristo_EnKF_nonlinear_LR_{}%obs_{}obsgap.csv".format(int(p * 100.), obs_gap))
    if p < 1:
        pd.DataFrame(errora_obs, columns=["RMSE_obs"]).to_csv("RMSE_hristo_EnKF_nonlinear_LR_observed_{}%obs_{}obsgap.csv".format(int(p * 100.), obs_gap))
        pd.DataFrame(errora_unobs, columns=["RMSE_unobs"]).to_csv("RMSE_hristo_EnKF_nonlinear_LR_unobserved_{}%obs_{}obsgap.csv".format(int(p * 100.), obs_gap))


