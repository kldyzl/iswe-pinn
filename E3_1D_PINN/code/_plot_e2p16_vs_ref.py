#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""_plot_e2p16_vs_ref.py —— 2D 全量（E2P 抛物床溃坝）**中槽截面** vs 1D 内蕴参考解，
                16 帧（t = 0, 0.1, ..., 1.5 s）4×4 网格对比图。
数据：PINN = E3_parabolic_dam/pinn/output/pinn_dam_parab_p1.npz（prof_h/prof_x/t_list）
      参考 = E3b_parabola_recheck/reference/output/e3b_n32000.npz（eta/x/t）
只读，不训练。
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "<workspace>/"
pinn = np.load(R + "E3_parabolic_dam/pinn/output/pinn_dam_parab_p1_snap.npz",
               allow_pickle=True)
ref = np.load(R + "E3b_parabola_recheck/reference/output/e3b_n32000.npz")

xp = np.asarray(pinn["prof_x"], float)          # 截面 x 网格（401）
hp = np.asarray(pinn["prof_h_snap"], float)     # [16, 401] 法向水深（中槽）
tp = np.asarray(pinn["t_snap"], float).ravel()
xr = np.asarray(ref["x"], float)
er = np.asarray(ref["eta"], float)              # [16, nx_ref]
tr = np.asarray(ref["t"], float).ravel()

ts = [round(0.1 * k, 1) for k in range(16)]
outdir = R + "E3_parabolic_dam/pinn/figures"
os.makedirs(outdir, exist_ok=True)

fig, axes = plt.subplots(4, 4, figsize=(8.8, 7.0), sharex=True, sharey=True)
for j, tv in enumerate(ts):
    ax = axes.flat[j]
    kp = int(np.argmin(np.abs(tp - tv)))
    kr = int(np.argmin(np.abs(tr - tv)))
    e_ref = er[kr]
    h_pin = np.interp(xr, xp, hp[kp])           # PINN 插到参考网格上
    ax.fill_between(xr, 0, e_ref, color="#B5D4F4", alpha=0.45, lw=0)
    ax.plot(xr, e_ref, color="#185FA5", lw=1.1, ls="--", label="reference (FV)")
    ax.plot(xr, h_pin, color="#C0392B", lw=1.3, label="2D PINN (mid-channel)")
    ax.axvline(2.0, color="0.65", ls=":", lw=0.7)
    ax.set_title("$t=%g$ s" % tv, fontsize=9.5, pad=2)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1.05)
    ax.tick_params(labelsize=7.5)
    if j == 0:
        ax.legend(fontsize=7, loc="upper right", framealpha=0.9)
for r in range(4):
    axes[r][0].set_ylabel(r"$\eta$ [m]", fontsize=9)
for c in range(4):
    axes[3][c].set_xlabel(r"$x^1$ [m]", fontsize=9)
fig.suptitle("E3 parabolic-bed dam break — 2D PINN mid-channel $\\eta$ (red) "
             "vs intrinsic 1D Godunov reference (blue dashed)", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.955))
p = os.path.join(outdir, "fig_e2p16_pinn_vs_ref_sections.png")
fig.savefig(p, dpi=155)
print("SAVED", p)

# 数值对照：逐帧 rel-L1（同一网格）
print("t     relL1    max|dh|")
for tv in ts:
    kp = int(np.argmin(np.abs(tp - tv)))
    kr = int(np.argmin(np.abs(tr - tv)))
    e_ref = er[kr]
    h_pin = np.interp(xr, xp, hp[kp])
    rel = np.sum(np.abs(h_pin - e_ref)) / np.sum(np.abs(e_ref))
    print("%.1f   %.4e  %.4f" % (tv, rel, np.abs(h_pin - e_ref).max()))
