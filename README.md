# Intrinsic shallow-water PINN: code for E1–E3

Reference implementation of the physics-informed neural network formulated on the
**geometrically intrinsic shallow water equations (ISWE)** over curved beds, together with
the intrinsic finite-volume solvers used to generate the reference solutions.

This repository contains **code only**. The computed fields, training logs, figures and the
data dictionary are archived on Zenodo:

> **Data: https://doi.org/10.5281/zenodo.23450658** (CC BY 4.0)

## Layout

| Directory | Content | Experiment |
|---|---|---|
| `E1_PINN/code/` | `iswe_pinn_monge.py`, `run_pinn_monge.sh` | Lake at rest on a doubly periodic bed |
| `E2_PINN/code/` | `iswe_pinn_e2_circulardam.py` | Circular dam break on the flat bed |
| `E2_FV/code/` | `swe_ref.py`, `compare_and_plot.py`, `plot_ref_e2.py` | Radially symmetric reference solution for E2 |
| `E3_1D_PINN/code/` | `swe_pinn_1d_v2.py`, `run_chain_v2.sh`, plotting scripts | Dam break on the parabolic bed, exactly reduced to 1D |
| `E3_2D_PINN/code/` | `iswe_pinn_e2p_parabola.py`, `_replot_surf3d.py` | Two-dimensional control run on the same bed |
| `E3_FV/code/` | `swe_ref.py`, `run_e3b_ref.py`, `compare_and_plot.py`, `plot_ref_e3b.py` | Intrinsic finite-volume references (1D and 2D) |

## Requirements

`requirements.txt` lists the versions used for the reported runs
(Python 3.11, NumPy, Matplotlib, PyTorch 2.6 with CUDA 12.4, SymPy). The PINN runs were
executed on a single NVIDIA Quadro RTX 6000 (24 GB) card; the two dam-break runs used one
card each and were run concurrently.

## Running

Each experiment directory contains a driver script:

```bash
python E1_PINN/code/iswe_pinn_monge.py        # or: bash E1_PINN/code/run_pinn_monge.sh
python E2_PINN/code/iswe_pinn_e2_circulardam.py
python E3_1D_PINN/code/swe_pinn_1d_v2.py      # or: bash E3_1D_PINN/code/run_chain_v2.sh
```

The finite-volume references are produced with `swe_ref.py` in the corresponding `*_FV`
directory. All scripts write to a local output directory; paths are relative, no
machine-specific location is hard-coded.

## What is where in the paper

The three configurations, the acceptance criteria and the reported metrics are described in
the *Numerical experiments* section of the manuscript. The metrics file names in the Zenodo
archive (`compare_E2.txt`, `rel_l1_vs_t.txt`, `wellbalance_audit.log`) match the tables of the
paper one to one.

## License

Both the code in this repository and the archived data and figures (Zenodo record above) are
released under the **Creative Commons Attribution 4.0 International** licence (CC BY 4.0);
see `LICENSE`.

## Citation

If you use this code, please cite the dataset and the manuscript:

```
Liu, D., He, S., Xie, X., & Liu, W. (2026). Numerical artefacts for the intrinsic
shallow-water PINN experiments (E1-E3): reference solutions, trained fields and diagnostic
logs (Version 1.0.0) [Dataset]. Zenodo. https://doi.org/10.5281/zenodo.23450658
```
