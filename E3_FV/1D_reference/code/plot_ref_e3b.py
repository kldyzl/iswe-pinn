#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""plot_ref_e3b.py — 参考解自己的「时间截面 / 快照网格 / 3D 曲面」

**只读 npz，不重算、不训练**。

产出（前 4 张为瀑布/总量图，后 2 张为**每时刻一格**的快照网格）：
  fig_e3b_ref_timesections.png   16 帧 η(x¹) 与 ζ(x¹) 叠加（viridis 按 t 着色）
  fig_e3b_ref_surface3d.png      ζ(x¹,t) 3D 曲面 + 床面（VE 仅作**显示层**纵向夸张）
  fig_e3b_ref_qout_mass.png      Q_out(t) 与 M(t)
  fig_e3b_ref_spacetime.png      η/ζ 时空热图
  fig_e3b_ref_snap_eta.png       **4×4 快照网格：每格一个时刻，水深 η 填色**
  fig_e3b_ref_snap_zeta.png      **4×4 快照网格：每格一个时刻，自由水面 ζ + 床面（水体填色）**

用法
----
  python plot_ref_e3b.py --npz ../output/e3b_n32000.npz --outdir ../figures
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

# VE 只体现在 3D 坐标盒的纵横比上（**不改数据、不误导色标**）。
VE = 2.0
# 快照网格配色（与论文其它图同一族，浅色填充 + 深色描边）
C_FILL = "#DCE9F7"
C_LINE = "#185FA5"
C_BED = "0.45"
# 每格画布（in）：可由 --panel 覆盖。**它决定纸面字号**：
# 纸面字号 = 源字号 × (插入宽度 / 整图宽)；画布越小 ⇒ 缩放越小 ⇒ 字越大。
PANEL = (1.55, 1.15)
SUFFIX = ""


def load(npz):
    d = np.load(npz)
    return {k: d[k] for k in d.files}


def bed_of(x):
    """E3 床 z(x¹) = (x¹ − 10)²/25（**仅用于画图**，不参与任何求解）。"""
    return (x - 10.0) ** 2 / 25.0


def fig_timesections(D, outdir):
    x, t, eta, zeta = D["x"], D["t"], D["eta"], D["zeta"]
    zb = bed_of(x)

    fig, ax = plt.subplots(1, 2, figsize=(11.0, 4.3), constrained_layout=True)
    norm = Normalize(vmin=float(t[0]), vmax=float(t[-1]))
    cmap = cm.viridis
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    for k in range(len(t)):
        c = cmap(norm(t[k]))
        ax[0].plot(x, eta[k], color=c, lw=1.25)
        ax[1].plot(x, zeta[k], color=c, lw=1.25)
    ax[1].plot(x, zb, color="0.35", lw=1.6, ls="--", label=r"bed $z$")

    ax[0].set_xlabel(r"$x^1$ [m]")
    ax[0].set_ylabel(r"normal depth $\eta$ [m]")
    ax[0].set_title(r"(a) reference $\eta(x^1,t)$")
    ax[1].set_xlabel(r"$x^1$ [m]")
    ax[1].set_ylabel(r"free surface $\zeta = z+\gamma\eta$ [m]")
    ax[1].set_title(r"(b) reference $\zeta(x^1,t)$")
    ax[1].legend(loc="upper right", fontsize=8, frameon=False)
    for a in ax:
        a.grid(alpha=0.25, lw=0.5)
        a.set_xlim(float(x[0]), float(x[-1]))
    cb = fig.colorbar(sm, ax=ax, shrink=0.9, pad=0.015)
    cb.set_label(r"$t$ [s]")

    p = os.path.join(outdir, "fig_e3b_ref_timesections.png")
    fig.savefig(p, dpi=200)
    plt.close(fig)
    print("  saved", p)


def _grid_axes(n, ncol=4):
    """快照网格画布。

    **面板尺寸取小**（1.55 × 1.15 in）是刻意的：整图从 6.2 in 缩到 \\textwidth(5.42 in)
    只缩 ≈0.75 倍，字号损失小；若沿用大画布（2.75 in/格）则要缩 0.4 倍，格内小标题
    在纸面上只剩 ~4 pt，完全看不清。
    """
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol,
                             figsize=(PANEL[0] * ncol, PANEL[1] * nrow),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    for k in range(n, len(axes)):
        axes[k].set_visible(False)
    return fig, axes, nrow, ncol


def fig_snap_eta(D, outdir):
    """4×4 快照网格：每格一个时刻，**水深 η 填色**（最易读的演化视图）。"""
    x, t, eta = D["x"], D["t"], D["eta"]
    n = len(t)
    fig, axes, nrow, ncol = _grid_axes(n, 4)
    ymax = float(eta.max()) * 1.08

    for k in range(n):
        ax = axes[k]
        ax.fill_between(x, 0.0, eta[k], color=C_FILL, lw=0, zorder=1)
        ax.plot(x, eta[k], color=C_LINE, lw=1.5, zorder=2)
        ax.axvline(2.0, color="0.6", lw=0.8, ls=":", zorder=0)
        ax.set_title(r"$t=%g$ s" % t[k], fontsize=11, pad=2)
        ax.grid(alpha=0.22, lw=0.5)
        ax.tick_params(labelsize=9, pad=0.5)
        ax.set_xlim(0.0, float(x[-1]))
        ax.set_ylim(0.0, ymax)

    for j in range(ncol):
        axes[(nrow - 1) * ncol + j].set_xlabel(r"$x^1$ [m]", fontsize=10)
    for i in range(nrow):
        axes[i * ncol].set_ylabel(r"$\eta$ [m]", fontsize=10)

    fig.suptitle(r"reference normal depth $\eta(x^1,t)$ — one panel per output instant",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    p = os.path.join(outdir, "fig_e3b_ref_snap_eta%s.png" % SUFFIX)
    fig.savefig(p, dpi=200)
    plt.close(fig)
    print("  saved", p)


def fig_snap_zeta(D, outdir):
    """4×4 快照网格：每格一个时刻，自由水面 ζ + 床面 z，**水体填色**（物理剖面）。"""
    x, t, zeta = D["x"], D["t"], D["zeta"]
    zb = bed_of(x)
    n = len(t)
    fig, axes, nrow, ncol = _grid_axes(n, 4)
    ymax = float(max(zeta.max(), zb.max())) * 1.06

    for k in range(n):
        ax = axes[k]
        ax.fill_between(x, zb, zeta[k], color=C_FILL, lw=0, zorder=1)
        ax.plot(x, zeta[k], color=C_LINE, lw=1.4, zorder=2,
                label=r"$\zeta$" if k == 0 else None)
        ax.plot(x, zb, color=C_BED, lw=1.2, ls="--", zorder=3,
                label=r"bed $z$" if k == 0 else None)
        ax.set_title(r"$t=%g$ s" % t[k], fontsize=11, pad=2)
        ax.grid(alpha=0.22, lw=0.5)
        ax.tick_params(labelsize=9, pad=0.5)
        ax.set_xlim(0.0, float(x[-1]))
        ax.set_ylim(0.0, ymax)
    axes[0].legend(loc="upper right", fontsize=9, frameon=False)

    for j in range(ncol):
        axes[(nrow - 1) * ncol + j].set_xlabel(r"$x^1$ [m]", fontsize=10)
    for i in range(nrow):
        axes[i * ncol].set_ylabel(r"$\zeta$ [m]", fontsize=10)

    fig.suptitle(r"reference free surface $\zeta = z+\gamma\eta$ and bed $z$ — "
                 r"one panel per output instant", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    p = os.path.join(outdir, "fig_e3b_ref_snap_zeta%s.png" % SUFFIX)
    fig.savefig(p, dpi=200)
    plt.close(fig)
    print("  saved", p)


def fig_surface3d(D, outdir):
    x, t, zeta = D["x"], D["t"], D["zeta"]
    zb = bed_of(x)
    X, T = np.meshgrid(x, t)

    fig = plt.figure(figsize=(8.6, 5.8), constrained_layout=True)
    ax = fig.add_subplot(projection="3d")
    surf = ax.plot_surface(X, T, zeta, cmap="viridis", rstride=1, cstride=1,
                           linewidth=0.0, antialiased=True, alpha=0.98)
    ax.plot(x, np.full_like(x, float(t[0])), zb, color="0.45", lw=1.1)
    ax.set_xlabel(r"$x^1$ [m]")
    ax.set_ylabel(r"$t$ [s]")
    ax.set_zlabel(r"$\zeta$ [m]")
    ax.set_title(r"reference free surface $\zeta(x^1,t)$  (VE$=%.1f$, display only)" % VE)
    ax.set_box_aspect((1.55, 1.0, 0.30 * VE))
    ax.view_init(elev=24, azim=-127)
    cb = fig.colorbar(surf, ax=ax, shrink=0.62, pad=0.08)
    cb.set_label(r"$\zeta$ [m]")

    p = os.path.join(outdir, "fig_e3b_ref_surface3d.png")
    fig.savefig(p, dpi=200)
    plt.close(fig)
    print("  saved", p)


def fig_qout_mass(D, outdir):
    t, q_out = D["t"], D["q_out"]
    tq, m_all = D["t_q"], D["m_all"]
    m0 = float(D["mass0"])

    fig, ax = plt.subplots(1, 2, figsize=(11.0, 4.0), constrained_layout=True)
    ax[0].plot(t, q_out[:len(t)], "o-", color="#1f77b4", lw=1.4, ms=4)
    ax[0].set_xlabel(r"$t$ [s]")
    ax[0].set_ylabel(r"$Q_{\rm out}(t)$ [m$^2$ s$^{-1}$]")
    ax[0].set_title(r"(a) outlet discharge")
    ax[0].grid(alpha=0.25, lw=0.5)

    ax[1].plot(tq, m_all, "-", color="#d62728", lw=1.5)
    ax[1].axhline(m0, color="0.4", ls="--", lw=1.0, label=r"$M(0)$")
    ax[1].set_xlabel(r"$t$ [s]")
    ax[1].set_ylabel(r"$M(t)$ [m$^3$]")
    ax[1].set_title(r"(b) water mass $\int \eta A\,dx$")
    ax[1].grid(alpha=0.25, lw=0.5)
    ax[1].legend(loc="best", fontsize=8, frameon=False)

    p = os.path.join(outdir, "fig_e3b_ref_qout_mass.png")
    fig.savefig(p, dpi=200)
    plt.close(fig)
    print("  saved", p)


def fig_spacetime(D, outdir):
    """时空热图：一眼看清演化（互补于 16 帧瀑布图）。"""
    x, t = D["x"], D["t"]
    fig, ax = plt.subplots(1, 2, figsize=(11.0, 4.2), constrained_layout=True)
    for j, (arr, lb, tag) in enumerate(((D["eta"], r"$\eta$", "a"),
                                        (D["zeta"], r"$\zeta$", "b"))):
        pcm = ax[j].pcolormesh(x, t, arr, shading="auto", cmap="viridis")
        ax[j].set_xlabel(r"$x^1$ [m]")
        ax[j].set_ylabel(r"$t$ [s]")
        ax[j].set_title(r"(%s) reference %s$(x^1,t)$" % (tag, lb))
        cb = fig.colorbar(pcm, ax=ax[j], pad=0.02)
        cb.set_label(lb + r" [m]")

    p = os.path.join(outdir, "fig_e3b_ref_spacetime.png")
    fig.savefig(p, dpi=200)
    plt.close(fig)
    print("  saved", p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--outdir", default="../figures")
    ap.add_argument("--only-snaps", action="store_true",
                    help="只出两张快照网格（迭代样图时用）")
    ap.add_argument("--panel", default=None,
                    help="每格画布 W,H（in），如 2.75,2.05；越大字号在纸面上越小")
    ap.add_argument("--suffix", default="", help="输出文件名后缀，便于并列比较各版本")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    global PANEL, SUFFIX
    if a.panel:
        w, h = a.panel.split(",")
        PANEL = (float(w), float(h))
    SUFFIX = a.suffix
    print("panel per cell = %s in ; suffix = %r" % (PANEL, SUFFIX))
    D = load(a.npz)
    print("npz keys:", sorted(D.keys()))
    if not a.only_snaps:
        fig_timesections(D, a.outdir)
        fig_surface3d(D, a.outdir)
        fig_spacetime(D, a.outdir)
        fig_qout_mass(D, a.outdir)
    fig_snap_eta(D, a.outdir)
    fig_snap_zeta(D, a.outdir)


if __name__ == "__main__":
    main()
