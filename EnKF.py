import numpy as np

def stochastic_enkf_analysis(x_ens, y_obs, y_ens, R, seed):
    """
    Corrected Stochastic EnKF with proper normalization and numerical stability

    Args:
        x_ens: Ensemble members (N x state_dim)
        y_obs: Observations (obs_dim)
        H: Observation function (returns (N x obs_dim) if vectorized, or callable)
        R: Observation error covariance (obs_dim x obs_dim)
    Returns:
        x_analysis: Updated ensemble (N x state_dim)
    """
    N = x_ens.shape[0]
    obs_dim = y_obs.shape[0]

    # 2. Center and normalize X and Y
    X = (x_ens - np.mean(x_ens, axis=0)) / np.sqrt(N - 1)  # (N x state_dim)
    Y = (y_ens - np.mean(y_ens, axis=0)) / np.sqrt(N - 1)  # (N x obs_dim)

    # 3. Compute Kalman gain with regularization
    YYT = Y.T @ Y + R # (obs_dim x obs_dim)
    try:
        inv_YYT = np.linalg.inv(YYT)  # Regularization
    except np.linalg.LinAlgError:
        inv_YYT = np.linalg.pinv(YYT)  # Fallback
    K = (X.T @ Y) @ inv_YYT  # (state_dim x obs_dim)

    # 4. Update ensemble
    rng = np.random.RandomState(seed)
    err = rng.multivariate_normal(np.zeros(obs_dim), R, size=N)
    y_osb_e = np.outer(np.ones(N), y_obs) + err
    """print("err",err)
    print("y_ens.shape",y_ens.shape)
    print("y_obs",y_obs.shape)
    print("K",K.shape)
    print(K)"""
    x_analysis = np.zeros((N, obs_dim))
    x_analysis = x_ens + (y_osb_e - y_ens) @ K.T

    return x_analysis