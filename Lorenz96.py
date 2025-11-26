### Import packages
import numpy as np
from Rev_SDE import REVERSE_SDE
from Rev_SDE_vanilla import REVERSE_SDE2
from EnKF_t import stochastic_enkf_analysis
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import torch

def lorenz96_step(x, dt, F):
    """
    Advance the Lorenz-96 system by one time step using RK4.

    Parameters
    ----------
    x : ndarray, shape (K,)
        Current state vector.
    dt : float
        Time step size.
    F : float
        Forcing term.

    Returns
    -------
    x_next : ndarray, shape (K,)
        State vector at next time step.
    """
    def tendency(x):
        K = len(x)
        xp1 = np.roll(x, -1)
        xm1 = np.roll(x, 1)
        xm2 = np.roll(x, 2)
        return (xp1 - xm2) * xm1 - x + F

    k1 = tendency(x)
    k2 = tendency(x + 0.5 * dt * k1)
    k3 = tendency(x + 0.5 * dt * k2)
    k4 = tendency(x + dt * k3)
    x_next = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    return x_next


def H(x):
    # Create the output vector
    if x.ndim == 2:
        #y = np.arctan(x)
        #y = np.array([x[:, 0], x[:, 1]]).T
        y = x
    elif x.ndim == 1:
        #y = np.arctan(x)
        #y = np.array([x[0], x[1]])
        y = x
    return y


# Setup
seed = 0
rng = np.random.RandomState(seed)
K = 40
F = 8.0
timestep = 1 + 200
dt = 0.01
x_truth = rng.normal(0, 1, size=K)
print("x0",x_truth)
x_choice = np.zeros((10000, len(x_truth)))
for i in range(10000):
    x_truth = lorenz96_step(x_truth, dt, F)                     ### nature run 10k steps
    x_choice[i] = x_truth
print("x_truth",x_truth)
obs_std = 0.1
obs_std_nonlinear = 0.01
obs_gap = 1
linear_regression = 0
par_obs = 1.
nobs = int(par_obs * K)
network = "Fixed_even"
choice_indxob = int(K/nobs)
#forward_error = 0.01
case = 3
### Case3: Nonlinear observation function
use_ensf = 1
use_enkf = 1 - use_ensf
N = 2000
rmse = []
rmse_obs = []
rmse_unobs = []




### Set up initial ensembles
x_ens = np.zeros((N,K))
x_ens_analysis = np.zeros((N,K))
x_ens = x_choice[rng.permutation(N)]
rmse_start = np.sqrt(((x_ens.mean(axis=0)-x_truth)**2).mean())
rmse.append(rmse_start)
print("rmse_start", rmse_start)

if use_ensf == True:
    if linear_regression == True and nobs < x_ens.shape[1]:
        mode = "ensf_LR"
    else:
        mode = "ensf_Vanilla"
elif use_enkf == True:
    if linear_regression == True and nobs < x_ens.shape[1]:
        mode = "enkf_LR"
    else:
        mode = "enkf_Vanilla"
if network == "Fixed_even":
    indxob = np.arange(0,K,choice_indxob)[0:nobs]
    if case == 3:
        percentage_arctan = 0.
    indx_indxob_linear = np.sort(rng.choice(np.arange(nobs), int(nobs * (1 - percentage_arctan)), replace=False),axis=None)
    indx_indxob_nonlinear = np.sort(np.setdiff1d(np.arange(nobs), indx_indxob_linear))
    indxunob = np.sort(np.setdiff1d(np.arange(x_truth.shape[0]), indxob))
    rmse_obs.append(np.sqrt(((x_ens[:,indxob].mean(axis=0)-x_truth[indxob])**2).mean()))
    rmse_unobs.append(np.sqrt(((x_ens[:,indxunob].mean(axis=0)-x_truth[indxunob])**2).mean()))

x_truth = lorenz96_step(x_truth, dt, F)
for j in range(N):
    x_ens[j, :] = lorenz96_step(x_ens[j, :], dt, F)
    #x_ens[j,:] = lorenz96_step(x_ens[j,:],dt,F) + rng.normal(loc=0,scale=forward_error * np.ones(K),size=K)



for i in np.arange(1, timestep):
    print("\n")
    print("*******  {}th time step **************".format(i))
    if i % obs_gap == 1 or obs_gap == 1:
        if network == "Random":
            indxob = np.sort(rng.choice(np.arange(K), size=nobs, replace=False))
            if case == 3:
                percentage_arctan = 1.
            indx_indxob_linear = np.sort(rng.choice(np.arange(nobs), int(nobs * (1 - percentage_arctan)), replace=False),axis=None)
            indx_indxob_nonlinear = np.sort(np.setdiff1d(np.arange(nobs), indx_indxob_linear))
            indxunob = np.sort(np.setdiff1d(np.arange(x_truth.shape[0]), indxob))

        """print("indxob", indxob)
        print("indx_indxob_linear", indx_indxob_linear)
        print("indx_indxob_nonlinear", indx_indxob_nonlinear)
        print("indxunob", indxunob)"""

        ### Create observations, dimension=(1000,nobs)
        obs_sigma = obs_std * np.ones(nobs)
        obs_sigma[indx_indxob_nonlinear] = obs_std_nonlinear
        #print("obs_sigma", obs_sigma)
        R = np.diag(obs_sigma ** 2)  # Observation covariance matrix used in EnKF
        obs_noise = rng.normal(loc=0, scale=obs_sigma, size=nobs)
        #print("obs_noise", obs_noise)
        y_obs = np.zeros(nobs)
        y_ens = np.zeros((N,nobs))
        y_obs = H(x_truth[indxob]) + obs_noise
        y_ens = H(x_ens[:, indxob])
        print("mode", mode)
        print("{}th pri_rmse=".format(i),np.sqrt(((x_ens.mean(axis=0) - x_truth) ** 2).mean()))
        #print("{}th rmse_obs=".format(i),np.sqrt(((x_ens[:, indxob].mean(axis=0) - x_truth[indxob]) ** 2).mean()))
        #print("{}th rmse_unobs=".format(i),np.sqrt(((x_ens[:, indxunob].mean(axis=0) - x_truth[indxunob]) ** 2).mean()))

        if use_ensf == True:
            print(1)
            user = REVERSE_SDE2(500, x_ens, N, y_obs, obs_sigma, nobs, indxob, indx_indxob_linear)
            x_ens_analysis = user.reverse_SDE()
        elif use_enkf == True:
            if linear_regression == 0:
                #print("x_truth", x_truth)
                #print("x_ens_prior", x_ens.mean(axis=0))
                #print("y_obs", y_obs)
                #print("x_ens_posterior", x_ens_analysis.mean(axis=0))
                #print("R",R)
                x_ens_analysis = stochastic_enkf_analysis(x_ens, y_obs, y_ens, R, seed)
            else:
                print(3)
                x_ens_subarray = x_ens[:, indxob].copy()
                x_ens_analysis[:, indxunob] = x_ens[:, indxunob].copy()
                x_ens_analysis_subarray = stochastic_enkf_analysis(x_ens_subarray, y_obs, y_ens, R, seed)
                x_ens_analysis[:, indxob] = x_ens_analysis_subarray
            seed += 1
        print("after update")
        #print("noise", obs_noise)
        #print("x_ens_prior_std", x_ens.std(axis=0))
        #print("x_ens_posterior_std", x_ens_analysis.std(axis=0))
        #print("y_obs", y_obs)
        #print("x_truth", x_truth[indxob])
        #print("tan(y_obs)", np.tan(y_obs))
        #print("x_ens_prior", x_ens.mean(axis=0))
        #pd.DataFrame(x_ens).to_csv("x_ens_lorenz96_118.csv")
        #print("x_ens_posterior", x_ens_analysis.mean(axis=0))
        #pd.DataFrame(x_ens_analysis).to_csv("x_ens_analysis_lorenz96_118.csv")



        ####### Step 2: apply Linear regression
        if linear_regression == True and nobs < x_ens.shape[1]:
            z_prior = H(x_ens[:,indxob])  ## shape (N,nobs)
            z_posterior = H(x_ens_analysis[:,indxob])  ## shape (N,nobs)
            dz = z_posterior - z_prior  ## shape (N,nobs)
            #print("dz", dz.mean(axis=0))
            z_prior_mean = z_prior.mean(axis=0)  ## shape (nobs,)
            z_posterior_mean = z_posterior.mean(axis=0)  ## shape (nobs,)
            #print("z_prior_mean", z_prior_mean)
            cov_zz = (z_prior - z_prior_mean).T @ (z_prior - z_prior_mean) / (N - 1)  ### shape (nobs,nobs)
            #print("Cov_zz", cov_zz, cov_zz.shape)
            #print("Condition number of cov_zz:", np.linalg.cond(cov_zz))
            if np.linalg.matrix_rank(cov_zz) == cov_zz.shape[0]:
                inv_cov_zz = np.linalg.inv(cov_zz)
            else:
                inv_cov_zz = np.linalg.pinv(cov_zz)  # fallback
            #print("inv_cov_zz", inv_cov_zz, inv_cov_zz.shape)
            x_prior_unobs_mean = x_ens[:, indxunob].mean(axis=0)  ## shape (K-nobs,1)
            cov_xz = (((x_ens[:, indxunob] - x_prior_unobs_mean).T @ (z_prior - z_prior_mean)) / (N - 1)).reshape(K-nobs, nobs)  # shape (K-nobs, nobs)
            #print("Cov_xz ", cov_xz, cov_xz.shape)
            ## update unobserved states of x_posterior:
            dx = dz @ (cov_xz @ inv_cov_zz).T  # shape (N,K-nobs)
            #print("cov", (cov_xz @ inv_cov_zz).T)
            print("x_ens_prior", x_ens.mean(axis=0))
            print("x_ens_posterior", x_ens_analysis.mean(axis=0))
            x_ens_analysis[:, indxunob] = x_ens[:, indxunob] + dx
            print("x_ens_posterior_LR", x_ens_analysis.mean(axis=0))
            #print("x_ens_posterior_LR_std", x_ens_analysis.std(axis=0))
        #print("after update")


    if i % obs_gap != 1 and obs_gap != 1:
        x_ens_analysis = x_ens.copy()

    ### Evaluate the performance of Filter by calculating RMSE.
    ### state space
    rmse_temp = np.sqrt(((x_ens_analysis.mean(axis=0) - x_truth) ** 2).mean())
    print("{}th pos_rmse=".format(i), rmse_temp)
    #print("{}th rmse_obs=".format(i), np.sqrt(((x_ens_analysis[:,indxob].mean(axis=0) - x_truth[indxob]) ** 2).mean()))
    #print("{}th rmse_unobs=".format(i), np.sqrt(((x_ens_analysis[:,indxunob].mean(axis=0) - x_truth[indxunob]) ** 2).mean()))
    rmse.append(rmse_temp)
    rmse_obs.append(np.sqrt(((x_ens_analysis[:,indxob].mean(axis=0) - x_truth[indxob]) ** 2).mean()))
    rmse_unobs.append(np.sqrt(((x_ens_analysis[:,indxunob].mean(axis=0) - x_truth[indxunob]) ** 2).mean()))

    x_truth = lorenz96_step(x_truth, dt, F)
    for j in range(N):
        x_ens[j, :] = lorenz96_step(x_ens_analysis[j, :], dt, F)
        #x_ens[j, :] = lorenz96_step(x_ens_analysis[j,:],dt,F) + rng.normal(loc=0,scale=forward_error * np.ones(K),size=K)


    ## Save the file:
    if i == timestep - 1:
        df = pd.DataFrame(rmse, columns=["RMSE_ensf"])
        df.to_csv("RMSE_{}_case{}_Lorenz96_{}size_{}{}_{}%obs_{}_{}_{}.csv".format(mode, case, N, network, nobs, int(nobs/K*100), int(percentage_arctan * 100), np.max(obs_sigma), obs_gap))
        pd.DataFrame(rmse_obs, columns=["RMSE_ensf"]).to_csv("RMSE_{}_observed_case{}_Lorenz96_{}size_{}{}_{}%obs_{}_{}_{}.csv".format(mode, case, N, network, nobs, int(nobs/K*100), int(percentage_arctan * 100), np.max(obs_sigma), obs_gap))
        pd.DataFrame(rmse_unobs, columns=["RMSE_ensf"]).to_csv("RMSE_{}_unobserved_case{}_Lorenz96_{}size_{}{}_{}%obs_{}_{}_{}.csv".format(mode, case, N, network, nobs, int(nobs/K*100), int(percentage_arctan * 100), np.max(obs_sigma), obs_gap))
        print("RMSE_{}_case{}_Lorenz96_{}size_{}{}_{}%obs_{}_{}_{}.csv".format(mode, case, N, network, nobs, int(nobs/K*100), int(percentage_arctan * 100), np.max(obs_sigma), obs_gap))
