#!/bin/bash
cd <workspace>/E3b_parabola_recheck/pinn1d/code
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=python3
REF=../../reference/output/e3b_n32000.npz
echo "[chain] R1 adam start $(date)"
$PY -u swe_pinn_1d_v2.py --device cuda --mode adam --iter 30000 --ref $REF --outdir output/v2_r1_adam
echo "[chain] R2a viscous start $(date)"
$PY -u swe_pinn_1d_v2.py --device cuda --mode viscous --iter 30000 --ref $REF --outdir output/v2_r2a_viscous
echo "[chain] R2b slabfv start $(date)"
$PY -u swe_pinn_1d_v2.py --device cuda --mode slabfv --iter 30000 --n-cell 128 --n-slab 30 --nq-x 32 --nq-t 6 --ref $REF --outdir output/v2_r2b_slabfv
echo "[chain] ALL DONE $(date)"
