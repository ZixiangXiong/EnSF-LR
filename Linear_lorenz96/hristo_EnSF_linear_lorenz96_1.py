import numpy as np
from scipy.integrate import odeint
import pandas as pd
from Rev_SDE_vanilla import REVERSE_SDE2
import sys


def main(seed):
    seed_sf = seed
    ### Define model
    Nx = 40;  # number of grid points along latitude circle
    F = 8;

    def lorenz96(x, t):  # right-hand side of the Lorenz 96 model
        n = len(x);
        dx = [(x[np.mod(i + 1, n)] - x[i - 2]) * x[i - 1] - x[i] + F for i in range(0, n)];
        # Q: Purpose of mod(i+1,n)?
        return dx

    x_truth = pd.read_csv("x_truth_trajectory_seed{}.csv".format(int(seed_sf))).iloc[:,1:].to_numpy().copy()  ## shape (500,40)
    x_initial_ensemble = pd.read_csv("X_initial_forecst_hristo_seed{}.csv".format(int(seed_sf))).iloc[:,1:].to_numpy().copy()  ## shape (Ne,40)
    y_obs_full = pd.read_csv("y_observation_trajectory_seed{}.csv".format(int(seed_sf))).iloc[:,1:].to_numpy().copy()  ## shape (500,40)

    ### Declarations
    rng = np.random.RandomState(42);  # to ensure results are reproducible
    time_step = 0.02;  # dt for numerical integration
    p = 0.25;  # fraction of the directly observed state variables
    obs_type = "fix_even"
    linear_regression = 1
    T = x_truth.shape[0];  # number of time steps to simulate (not the absolute time)
    errorf_k = np.zeros(T);  # array to store the forecast errors
    errora_k = []  # array to store the analysis errors
    errora_obs = []
    errora_unobs = []
    spread = []
    x_0 = np.zeros(shape=(505, 2000))
    x_1 = np.zeros(shape=(505, 2000))
    x_0_forecast = np.zeros(shape=(505, 2000))
    x_1_forecast = np.zeros(shape=(505, 2000))


    I = np.eye(Nx, Nx);  # identity matrix
    Ne = x_initial_ensemble.shape[0];  # the ensemble size we will use in this experiment
    obs_gap = 10
    ones = np.ones(Ne);  # a vector of 1s, to be used later in creating
    # the full ensemble matrices
    loc = 3;  # localization radius to be used in this experiment
    infl = 1.03;  # inflation factor to be used in this experiment
    ### Ensemble size

    ### Initial true state and ensemble
    Xf_k = x_initial_ensemble.copy().T;  ## shape (40, 2000)
    Xa_k = np.zeros_like(Xf_k)
    errora_k.append(np.sqrt(((Xf_k.mean(axis=1) - x_truth[0, :]) ** 2).mean()))

    ### Observation-related variables
    Ny = int(round(p * Nx));  # number of observations
    sig_obs = 0.1;  # ob error std
    obs_sigma = sig_obs * np.ones(Ny)
    R_k = (sig_obs ** 2) * np.eye(Ny, Ny);  # ob error covariance matrice
    if obs_type == "fix_even":
        # Observation at time level k
        obs_comp = np.arange(0, Nx, int(1 / p))[0:Ny]
        unobs_comp = np.sort(np.setdiff1d(np.arange(Nx), obs_comp))
    # print("obs_comp",obs_comp)
    # print("unobs_comp",unobs_comp)

    ### Loop over time and assimilate observations (when present)
    for k in range(0, T):
        print("\n")
        print(f'Assimilation cycle: {k}');

        # Forecast mean
        xf_k = np.mean(Xf_k, 1);

        # (Localized) forecast covariance
        # L = create_localization_matrix(loc, Nx);
        # RMSE of forecast mean
        # errorf_k[k] = np.linalg.norm(xf_k - xt_k);
        errorf_k[k] = np.sqrt(((xf_k - x_truth[k, :]) ** 2).mean())
        print("forecast error=", errorf_k[k])
        print("obs error forecast=", np.sqrt(((Xf_k.mean(axis=1)[obs_comp] - x_truth[k, obs_comp]) ** 2).mean()))
        print("unobs error forecast=", np.sqrt(((Xf_k.mean(axis=1)[unobs_comp] - x_truth[k, unobs_comp]) ** 2).mean()))

        if k % obs_gap == 0:
            print("DA will be done")
            if linear_regression == 0:
                H_k = I[obs_comp, :];  # ob operator
                y_k = y_obs_full[k, obs_comp];  # true observation value
                # Preprocess the data for EnSF
                x_ens = Xf_k.copy().T
                N = Ne
                y_obs = y_k.copy()
                nobs = Ny
                indxob = obs_comp.copy()
                indx_indxob_linear = np.array(np.arange(Ny))
                # EnSF update
                user = REVERSE_SDE2(500, x_ens, N, y_obs, obs_sigma, nobs, indxob, indx_indxob_linear)
                x_ens_analysis = user.reverse_SDE()
                Xa_k = x_ens_analysis.copy().T
            if linear_regression == 1 and Ny < Nx:
                ### First step: update on observed states via EnSF
                H_k = I[obs_comp, :];  # ob operator
                y_k = y_obs_full[k, obs_comp];  # true observation value
                # Preprocess the data for EnSF
                x_ens = Xf_k.copy().T
                N = Ne
                y_obs = y_k.copy()
                nobs = Ny
                indxob = obs_comp.copy()
                indx_indxob_linear = np.array(np.arange(Ny))
                # EnSF update
                user = REVERSE_SDE2(500, x_ens, N, y_obs, obs_sigma, nobs, indxob, indx_indxob_linear)
                x_ens_analysis = user.reverse_SDE()
                Xa_k = x_ens_analysis.copy().T
                print("obs_analysis error pre=",
                      np.sqrt(((Xa_k.mean(axis=1)[obs_comp] - x_truth[k, obs_comp]) ** 2).mean()))
                print("unobs_analysis error pre=",
                      np.sqrt(((Xa_k.mean(axis=1)[unobs_comp] - x_truth[k, unobs_comp]) ** 2).mean()))

                ### Second step: update on unobserved states via Linear Regression
                # Compute the increment of observed ensembles in model space
                X_obs_increment = Xa_k[obs_comp] - Xf_k[obs_comp]
                # Linear regression update
                print("LR will be done")
                Xf_k_obs_center = (Xf_k[obs_comp, :] - Xf_k[obs_comp, :].mean(axis=1, keepdims=True)) / np.sqrt(
                    Ne - 1)  # shape (Ny,Ne)
                cov_zz = Xf_k_obs_center @ Xf_k_obs_center.T  # shape (Ny,Ny)
                if np.linalg.matrix_rank(cov_zz) == cov_zz.shape[0]:
                    inv_cov_zz = np.linalg.inv(cov_zz)
                else:
                    inv_cov_zz = np.linalg.pinv(cov_zz)  # fallback
                Xf_k_unobs_center = (Xf_k[unobs_comp, :] - Xf_k[unobs_comp, :].mean(axis=1, keepdims=True)) / np.sqrt(
                    Ne - 1)  # shape (Nx-Ny,Ne)
                cov_xz = (Xf_k_unobs_center @ Xf_k_obs_center.T).reshape(Nx - Ny, Ny)  # shape (Nx-Ny,Ny)
                ## Update unobserved states of Xa_k:
                dx = cov_xz @ np.linalg.solve(cov_zz, X_obs_increment)
                Xa_k[unobs_comp, :] = Xf_k[unobs_comp, :] + dx  ### shape (Nx-Ny,Ne)
        else:
            Xa_k = Xf_k.copy()

        x_0[k, :] = Xa_k[0, :].copy()
        x_1[k, :] = Xa_k[1, :].copy()
        x_0_forecast[k, :] = Xf_k[0, :].copy()
        x_1_forecast[k, :] = Xf_k[1, :].copy()



        if k == T - 1:
            pd.DataFrame(x_0).to_csv("x_0_traj_linear_hristo_EnSF_LR_lorenz96_{}%obs_{}obsgap_seed{}.csv".format(int(p * 100.), obs_gap, seed_sf))
            pd.DataFrame(x_1).to_csv("x_1_traj_linear_hristo_EnSF_LR_lorenz96_{}%obs_{}obsgap_seed{}.csv".format(int(p * 100.), obs_gap, seed_sf))
            pd.DataFrame(x_0_forecast).to_csv("x_0_traj_linear_forecast_hristo_EnSF_LR_lorenz96_{}%obs_{}obsgap_seed{}.csv".format(int(p * 100.), obs_gap, seed_sf))
            pd.DataFrame(x_1_forecast).to_csv("x_1_traj_linear_forecast_hristo_EnSF_LR_lorenz96_{}%obs_{}obsgap_seed{}.csv".format(int(p * 100.), obs_gap, seed_sf))


        # RMSE of analysis mean
        # 2. Ensemble spread per state variable
        spread_per_var = np.std(Xa_k, axis=1)  # [Nx]
        mean_spread = spread_per_var.mean()
        spread.append(mean_spread)
        print(f"Mean ensemble spread: {mean_spread:.4f}")

        xa_k = Xa_k.mean(axis=1)
        rmse = np.sqrt(((xa_k - x_truth[k, :]) ** 2).mean())
        errora_k.append(rmse)
        spread_skill_ratio = mean_spread / rmse
        print("Spread / RMSE = {:.2f}".format(spread_skill_ratio))

        errora_obs.append(np.sqrt(((xa_k[obs_comp] - x_truth[k, obs_comp]) ** 2).mean()))
        errora_unobs.append(np.sqrt(((xa_k[unobs_comp] - x_truth[k, unobs_comp]) ** 2).mean()))
        print("analysis error=", errora_k[-1])
        print("obs_analysis error=", errora_obs[-1])
        print("unobs_analysis error=", errora_unobs[-1])

        # Forecast to next time (both ensemble and nature run)
        for e in range(0, Ne):
            Xa_e_S = odeint(lorenz96, Xa_k[:, e], [0, time_step]);
            Xf_k[:, e] = Xa_e_S[-1, :]

    if linear_regression == 0:
        pd.DataFrame(errora_k, columns=["RMSE_ensf"]).to_csv("RMSE_lorenz96_linear_hristo_EnSF_{}%obs_{}obsgap_seed{}.csv".format(int(p * 100.), obs_gap, seed_sf))
        if p < 1:
            pd.DataFrame(errora_obs, columns=["RMSE_obs"]).to_csv("RMSE_lorenz96_linear_hristo_EnSF_observed_{}%obs_{}obsgap_seed{}.csv".format(int(p * 100.), obs_gap,seed_sf))
            pd.DataFrame(errora_unobs, columns=["RMSE_unobs"]).to_csv("RMSE_lorenz96_linear_hristo_EnSF_unobserved_{}%obs_{}obsgap_seed{}.csv".format(int(p * 100.), obs_gap,seed_sf))
    elif linear_regression == 1:
        pd.DataFrame(errora_k, columns=["RMSE_ensf"]).to_csv("RMSE_lorenz96_linear_hristo_EnSF_LR_{}%obs_{}obsgap_seed{}.csv".format(int(p * 100.), obs_gap, seed_sf))
        if p < 1:
            pd.DataFrame(errora_obs, columns=["RMSE_obs"]).to_csv("RMSE_lorenz96_linear_hristo_EnSF_LR_observed_{}%obs_{}obsgap_seed{}.csv".format(int(p * 100.), obs_gap,seed_sf))
            pd.DataFrame(errora_unobs, columns=["RMSE_unobs"]).to_csv("RMSE_lorenz96_linear_hristo_EnSF_LR_unobserved_{}%obs_{}obsgap_seed{}.csv".format(int(p * 100.),obs_gap, seed_sf))


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    main(seed)
