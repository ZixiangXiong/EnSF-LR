# EnSF-LR

Experiments for ensemble score filtering with likelihood-regulated reverse SDE updates.

## Repository Structure

```text
EnSF-LR/
├── Linear_lorenz63/
│   ├── data/
│   ├── figures/
│   └── *.py
├── Linear_lorenz96/
│   ├── data/
│   ├── figures/
│   └── *.py
├── Nonlinear_lorenz63/
│   ├── data/
│   ├── figures/
│   └── *.py
├── Nonlinear_lorenz96/
│   ├── data/
│   ├── figures/
│   └── *.py
├── Rev_SDE_vanilla.py
└── README.md
```

## Folders

- `Linear_lorenz63/`: Lorenz-63 experiments with linear observation operators.
- `Linear_lorenz96/`: Lorenz-96 experiments with linear observation operators.
- `Nonlinear_lorenz63/`: Lorenz-63 experiments with nonlinear observation operators.
- `Nonlinear_lorenz96/`: Lorenz-96 experiments with nonlinear observation operators.
- `data/`: Generated or prepared experiment data for each setting.
- `figures/`: Output plots and diagnostics for each setting.
- `Rev_SDE_vanilla.py`: Shared reverse SDE implementation.

## Notes

The empty `data/` and `figures/` directories contain `.gitkeep` files so the directory layout is preserved in Git.
