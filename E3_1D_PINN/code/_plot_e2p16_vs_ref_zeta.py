#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""_plot_e2p16_vs_ref_zeta.py —— 2D 全量（E2P）**竖直方向**：自由水面高程
        ζ = z(x¹) + γ(x¹)·η 的中槽截面 vs 1D 内蕴参考解，16 帧 4×4。
        每帧另画床面 z（灰实线）+ 参考水体填充。只读 npz，不训练。"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "<workspace>/"
pinn = np.load(R + "E3_parabolic_dam/pinn/output/pinn_dam_parab_p1_snap.npz",
               allow_pickle=True)
ref = np.load(R + "E3b_parabola_recheck/reference/output/e3b_n32000.npz")

xp = np.asarray(pinn["prof_x"], float)
zp = np.asarray(pinn["prof_zeta_snap"], float)      # [16,401] ζ（竖直高程）
tp = np.asarray(pinn["t_snap"], float).ravel()
xr = np.asarray(ref["x"], float)
zr = np.asarray(ref["zeta"], float)                 # [16, nx_ref]
tr = np.asarray(ref["t"], float).ravel()
bed = (xr - 10.0) ** 2 / 25.0                       # z = (x−10)²/25

ts = [round(0.1 * k, 1) for k in range(16)]
outdir = R + "E3_parabolic_dam/pinn/figures"
fig, axes = plt.subplots(4, 4, figsize=(8.8, 7.0), sharex=True, sharey=True)
for j, tv in enumerate(ts):
    ax = axes.flat[j]
    kp = int(np.argmin(np.abs(tp - tv)))
    kr = int(np.argmin(np.abs(tr - tv)))
    z_ref = zr[kr]
    z_pin = np.interp(xr, xp, zp[kp])
    ax.fill_between(xr, bed, z_ref, color="#B5D4F4", alpha=0.45, lw=0)
    ax.plot(xr, bed, color="0.45", lw=0.9)
    ax.plot(xr, z_ref, color="#185FA5", lw=1.1, ls="--", label="reference (FV)")
    ax.plot(xr, z_pin, color="#C0392B", lw=1.3, label="2D PINN (mid-channel)")
    ax.axvline(2.0, color="0.65", ls=":", lw=0.7)
    ax.set_title("$t=%g$ s" % tv, fontsize=9.5, pad=2)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.7)
    ax.tick_params(labelsize=7.5)
    if j == 0:
        ax.legend(fontsize=6.8, loc="upper right", framealpha=0.9)
for r in range(4):
    axes[r][0].set_ylabel(r"$\zeta$ [m]", fontsize=9)
for c in range(4):
    axes[3][c].set_xlabel(r"$x^1$ [m]", fontsize=9)
fig.suptitle("E3 parabolic-bed dam break — vertical water surface "
             "$\\zeta=z+\\gamma\\eta$: 2D PINN (red) vs 1D reference (blue dashed) ; bed $z$ (grey)",
             fontsize=10.5)
fig.tight_layout(rect=(0, 0, 1, 0.955))
p = os.path.join(outdir, "fig_e2p16_pinn_vs_ref_sections_zeta.png")
fig.savefig(p, dpi=155)
print("SAVED", p)
print("t     relL1(zeta)  max|dzeta|  cos-match(cosθ)")
for tv in ts:
    kp = int(np.argmin(np.abs(tp - tv)))
    kr = int(np.argmin(np.abs(tr - tv)))
    z_ref = zr[kr]
    z_pin = np.interp(xr, xp, zp[kp])
    rel = np.sum(np.abs(z_pin - z_ref)) / np.sum(np.abs(z_ref))
    print("%.1f   %.4e   %.4f" % (tv, rel, np.abs(z_pin - z_ref).max()))
