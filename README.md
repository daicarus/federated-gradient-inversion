# Beyond Idealized Settings: Gradient Inversion Attacks and Defense Evaluation in Non-IID Cross-Silo Federated Learning

Final year design project (FYDP), United International University.

We evaluate two optimization-based gradient-inversion attacks — Deep Leakage from
Gradients (DLG) and Inverting Gradients (IG) — and three defenses (gradient pruning,
differential privacy, and their combination) under realistic non-IID cross-silo
federated learning, rather than the idealized conditions common in prior work.

## Key findings

- Reconstruction quality is not fixed. It rises as the model trains, and falls as
  batch size and the number of local steps grow.
- DLG (squared-L2 loss) is sensitive to gradient **magnitude**; IG (cosine loss) is
  sensitive only to gradient **direction**. This single difference explains the
  batch-size crossover, IG's flat behaviour across multi-step updates, and the
  defense results.
- **False security:** gradient clipping and light noise change gradient magnitude but
  not direction, so they break DLG while IG bypasses them entirely. Only defenses that
  corrupt gradient direction (strong noise, heavy pruning, combined) stop IG.
- On the privacy–utility trade-off (at convergence), pruning with error feedback and
  the combined defense suppress IG at almost no accuracy cost, while differential
  privacy alone pays a real accuracy penalty for the same protection. The combined
  defense is competitive but not a clear improvement over pruning alone.

## Repository structure

```
FYDP_beyond_idealized.ipynb   Main notebook (shared library + all experiments)
requirements.txt              Python dependencies
figures/                      Result figures (axes, defense matrix, reconstruction grids)
results/                      Optional: saved numeric outputs
```

## How to run

Designed for a single GPU (developed on an NVIDIA T4 via Kaggle).

1. Open the notebook. Set the accelerator to GPU.
2. Run the **shared library** cell first.
3. Run **Step 1** three times, changing `WHICH`: `"smoke"`, then `"noniid"`, then `"iid"`.
   (Both `noniid` and `iid` are needed for the IID-vs-non-IID comparison.)
4. Run Steps 2–12 in order. Do not use "Run All": Step 1 must be run three times, and
   the long training cells should be run individually.

Each cell saves its figures to `outputs/` and prints its result tables.

## Method summary

- Dataset: CIFAR-10, pixels in [0, 1]. Non-IID split via Dirichlet (alpha = 0.5);
  near-IID via alpha = 100. Cross-silo, 10 clients.
- Model: small 3-conv sigmoid CNN (~15.8k params), as in the original DLG work.
- Training: FedAvg, 60 rounds, 2 local epochs, SGD lr = 0.1.
- Metrics: PSNR, SSIM (primary), MSE; model test accuracy for utility.
- All reconstruction numbers are means over 5 victim inputs (mean +/- std).

## References

- Zhu, Liu, Han. Deep Leakage from Gradients. NeurIPS 2019.
- Zhao, Mopuri, Bilen. iDLG: Improved Deep Leakage from Gradients. 2020.
- Geiping et al. Inverting Gradients — How Easy Is It to Break Privacy in
  Federated Learning? NeurIPS 2020.

## Notes

This is coursework. The combined pruning + differential-privacy defense is not novel;
prior work has studied it. The contribution here is the evaluation under realistic
conditions and the comparison of attacks and defenses, not the defense mechanism.
