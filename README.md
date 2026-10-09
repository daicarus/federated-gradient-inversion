# Beyond Idealized Settings

Gradient inversion attacks and defense evaluation in non-IID cross-silo federated
learning. Final year design project, United International University.

Two attacks (DLG, Inverting Gradients) against FedAvg under Dirichlet non-IID
partitioning, on CIFAR-10 and PathMNIST, with gradient clipping, Gaussian noise,
pruning, and combinations as defenses.

## Setup

10 cross-silo clients, Dirichlet alpha = 0.5, 60 rounds, 2 local epochs,
batch 64, SGD lr = 0.1, seed 42. Model is a 3-conv sigmoid CNN, 15,057 params.
PathMNIST final accuracy 0.6077.

DLG: L-BFGS, squared-L2, 300 iterations. IG: Adam, cosine distance plus TV prior
0.01, 1200 iterations. Both with 2 restarts and iDLG analytic label recovery.
Attack latents are drawn from a generator seeded on (victim, restart) only, so
results are comparable across defense conditions.

## Results

Null floor: SSIM of three non-attacks scored against each victim, PathMNIST.

| | SSIM |
|---|---|
| Uniform noise | 0.011 |
| Another victim's image | 0.246 |
| Grey square | 0.395 |
| Dataset mean image | 0.402 |
| DLG, undefended | 0.145 |
| IG, undefended | 0.246 |

Neither attack clears its own victim's floor in most cases (DLG 3/10, IG 4/10).
On CIFAR-10 both clear it (DLG 0.741, IG 0.503).

Defense matrix, PathMNIST SSIM with 95% CI over 10 victims.

| Defense | DLG | IG |
|---|---|---|
| none | 0.145 ± 0.073 | 0.246 ± 0.163 |
| clip C=4 | 0.058 ± 0.056 | 0.246 ± 0.163 |
| noise sigma=0.01 | 0.023 ± 0.018 | 0.204 ± 0.123 |
| noise sigma=0.1 | 0.030 ± 0.018 | 0.101 ± 0.048 |
| prune 90% | 0.047 ± 0.033 | 0.132 ± 0.063 |
| combined | 0.042 ± 0.026 | 0.135 ± 0.059 |

Clipping cuts DLG by 60% and leaves IG unchanged (paired difference
-0.000 ± 0.000). A rescaling probe over a 100x gradient-scale range gives IG
0.0790-0.0791 flat while DLG peaks at scale 1.0 and drops either side. IG's
cosine objective reads gradient direction only, so clipping cannot affect it.

Linkability: each reconstruction scored against all 10 victims, PathMNIST
undefended.

| Attack | SSIM | Mean rank | Top-1 | Matched/mismatched |
|---|---|---|---|---|
| DLG | 0.145 | 2.20 | 7/10 | 4.72x |
| IG | 0.246 | 3.70 | 2/10 | 1.62x |

DLG scores lower SSIM and identifies more victims. IG's TV prior produces texture
resembling every victim, giving it mismatched SSIM 0.152 against DLG's 0.031. On
CIFAR-10 both saturate at rank 1.00, 10/10.

PathMNIST class 1 is `background`. Two of ten randomly drawn victims were blank
tiles, against which a grey square scores 0.852 and 0.897. Per-victim floors span
0.141 to 0.897.

## Layout

```
notebooks/
  federated-gradient-inversion.ipynb   CIFAR-10, four axes of realism
  medmnist_replication.ipynb           PathMNIST, null floor, linkability
  cifar_control.ipynb                  CIFAR-10 control for the above
scripts/ptload.py                      read the .pt files without torch
results/                               JSON, reconstructions, cross-SSIM matrices
figures/                               plots for both datasets
```

## Running

Single GPU, developed on a Kaggle T4.

`federated-gradient-inversion.ipynb`: run the shared library cell, then Step 1
three times with `WHICH` set to `"smoke"`, `"noniid"`, `"iid"`, then Steps 2-12
in order. Do not use Run All.

`medmnist_replication.ipynb`: run Cell S with `SMOKE = True` first, then set
`SMOKE = False` and run M1, N, X, M2R, M2P, I, A, M2b, M7. Run X (null baseline)
before M2R (defense matrix). M2R checkpoints after each condition; the matrix
takes about two hours.

## Reading the results

```python
import sys; sys.path.insert(0, "scripts")
from ptload import load

d = load("results/med_recons.pt")
d["all_recon"]["clip C=4|IG"]   # (10, 3, 32, 32)
d["VX"], d["VY"]                # victim images and labels
```

```python
import numpy as np
z = np.load("results/med_cross_ssim.npz")
z["none__DLG"]                  # [i, j] = SSIM(recon_i, victim_j)
```

`results/` holds the only copies of the per-victim data and is committed
deliberately, so the `*.pt` rule in `.gitignore` is scoped to `outputs/` and
`data/`.

## Limitations

One image per attack, not a batch. 15k-parameter CNN only. 95% CIs over 10
victims are wide; per-victim arrays are in `results_medmnist.json` and the paired
differences are more reliable than comparing means. `results_cifar.json` was lost
to a Kaggle session reset, but the CIFAR numbers above were recomputed from
`cifar_recons.pt` and `cifar_cross_ssim.npz`.

The combined pruning and DP defense is not novel; prior work has studied it. The
clipping bypass is in Li et al. (CVPR 2022) with the same parameters, and
magnitude invariance is stated in Geiping et al.

## References

- Zhu, Liu, Han. Deep Leakage from Gradients. NeurIPS 2019.
- Zhao, Mopuri, Bilen. iDLG: Improved Deep Leakage from Gradients. 2020.
- Geiping et al. Inverting Gradients. NeurIPS 2020.
- Yang et al. MedMNIST v2. Scientific Data, 2023.
