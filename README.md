# EnSF-LR

Research code for **A Two-Step Ensemble Score Filter for Data Assimilation in Partially Observed Systems**.

This project implements and tests an Ensemble Score Filter with Linear Regression (EnSF-LR), a data assimilation method for estimating the full state of nonlinear dynamical systems when only part of the state is observed.

## Project Summary

Many forecasting problems, including weather and climate applications, require estimating hidden system variables from sparse noisy observations. Standard ensemble methods transfer information through covariance, while score-based methods can capture nonlinear likelihood effects but may only directly update observed variables.

EnSF-LR combines both ideas:

1. **Score-based observed-state update**: apply an Ensemble Score Filter update to the observed components using a reverse-SDE formulation.
2. **Regression-based information transfer**: use ensemble-estimated linear regression to transfer the observed-state correction to unobserved variables.

The experiments compare EnSF-LR with Ensemble Score Filter (EnSF) and Ensemble Kalman Filter (EnKF) baselines on Lorenz-63 and Lorenz-96 systems.

## What This Demonstrates

- Data assimilation for partially observed nonlinear systems
- Ensemble filtering and stochastic simulation
- Reverse-SDE score-based analysis updates
- Linear-regression information transfer between observed and unobserved variables
- Numerical experiment design with Lorenz-63 and Lorenz-96 models
- Python scientific computing with NumPy, SciPy, pandas, and PyTorch

## Repository Structure

```text
EnSF-LR/
├── Linear_lorenz63/
│   ├── figures/
│   ├── generate_data_linear_lorenz63.py
│   ├── EnSF_linear_lorenz63_*.py
│   └── EnKF_linear_lorenz63_*.py
├── Linear_lorenz96/
│   ├── figures/
│   ├── generate_data_linear_lorenz96.py
│   ├── EnSF_linear_lorenz96_*.py
│   └── EnKF_linear_lorenz96_*.py
├── Nonlinear_lorenz63/
│   ├── figures/
│   ├── generate_data_lorenz63.py
│   ├── EnSF_nonlinear_lorenz63_*.py
│   └── EnKF_nonlinear_lorenz63_*.py
├── Nonlinear_lorenz96/
│   ├── figures/
│   ├── generate_data_nonlinear_lorenz96.py
│   ├── EnSF_nonlinear_lorenz96_*.py
│   └── EnKF_nonlinear_lorenz96_*.py
├── Rev_SDE_vanilla.py
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

## Experiment Groups

- `Linear_lorenz63/`: Lorenz-63 with sparse linear observations.
- `Linear_lorenz96/`: Lorenz-96 with sparse linear observations.
- `Nonlinear_lorenz63/`: Lorenz-63 with sparse nonlinear observations.
- `Nonlinear_lorenz96/`: Lorenz-96 with sparse nonlinear observations.
- `Rev_SDE_vanilla.py`: Shared reverse-SDE implementation used by EnSF experiments.

The generated CSV data and result files are intentionally not stored in the repository.

## Installation

Create a Python environment and install dependencies:

```bash
pip install -r requirements.txt
```

For Intel macOS environments using `torch==2.2.2`, keep NumPy below version 2:

```bash
pip install numpy==1.26.4 torch==2.2.2
```

## Quick Example

Run one nonlinear Lorenz-63 experiment:

```bash
cd Nonlinear_lorenz63
python generate_data_lorenz63.py 50
python EnSF_nonlinear_lorenz63_0.py 50
python EnKF_nonlinear_lorenz63_0.py 50
```

The optional numeric argument is the random seed.

## Method Overview

EnSF-LR uses a two-step analysis update for partially observed systems. First, a score-based reverse-SDE update is applied to the observed state variables. Second, the observed-state analysis increments are transferred to the unobserved state variables using ensemble-estimated linear regression. The method is evaluated on Lorenz-63 and Lorenz-96 systems under sparse linear and nonlinear observations and compared with EnSF and EnKF baselines.

## Citation

If you use this code, please cite:

```bibtex
@article{xiong2026ensflr,
  title={A Two-Step Ensemble Score Filter for Data Assimilation in Partially Observed Nonlinear Systems},
  author={Xiong, Zixiang and Bao, Feng and Chipilski, Hristo G. and Liang, Siming and Tang, Jingqiao and Zhang, Guannan},
  year={2026}
}
```
