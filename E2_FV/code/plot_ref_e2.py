#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""plot_ref_e2.py -- T1：E2 参考解自己的「时间截面 / 快照网格 / 3D 快照 / 前锋」

**只读 npz，不重算、不训练**。参考解由 swe_ref.py 的径向（一维守恒 FV + HLL）
求解器给出；床为平底 z ≡ 0，故 ζ = z + γ h = h（γ ≡ 1）。

产出（落在 --outdir）：
  fig_e2_ref_timesections.png   (a) 径向水深 h(r) 的 11 帧叠加（viridis 按 t 着色）
                                (b) 激波锋 r_f(t) 与 0.1 s ≤ t ≤ 1 s 的线性拟合
  fig_e2_ref_surface3d.png      ζ(x¹,x²) 在 [-10,10]² 上的 3D 快照总览（Blues）
  fig_e2_ref_snap_grid.png      **3×4 快照网格：每格一个时刻，水深 h(r) 填色**
  fig_e2_ref_front.png          激波锋 r_f(t) 单图（供与快照网格上下拼版用）

用法
----
  python plot_ref_e2.py --npz ../output/e2_n8000.npz --outdir ../figures
  python plot_ref_e2.py --npz ../output/e2_n8000.npz --outdir ../figures --only-snaps
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

# --- 与 swe_ref.py / compare_and_plot.py 一致的常数 ---------------------------
HALF = 10.0                       # 方形域 [-10,10]^2 的半边长
R0 = 3.0                          # 初始坝半径
S_PLANAR = 4.183127921958328      # 同一数据 (hL=2, hR=1) 的平面精确激波速度
C_FILL = "#DCE9F7"
C_LINE = "#185FA5"


def load(p):
    d = np.load(p, allow_pickle=True)
    return {k: d[k] for k in d.files}


def _decim(n, target):
    return max(1, int(n) // int(target))


def _grid_axes(n, ncol=4):
    nrow = int(np.ceil(n / ncol))
    # 面板取小（1.55 x 1.15 in）：整图缩到 \textwidth 时只缩 ~0.7 倍，
    # 格内小标题在纸面上约 8pt；若用大画布会缩到 ~4pt 而看不清。
    fig, axes = plt.subplots(nrow, ncol, figsize=(1.55 * ncol, 1.15 * nrow),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    for k in range(n, len(axes)):
        axes[k].set_visible(False)
    return fig, axes, nrow, ncol


def fig_snap_grid(D, outdir):
    """3×4 快照网格：每格一个时刻，水深 h(r) 填色（比 11 条叠线清楚得多）。"""
    r, t, h = D["r"], D["t"], D["h"]
    n = len(t)
    fig, axes, nrow, ncol = _grid_axes(n, 4)
    ymax = float(h.max()) * 1.08

    for k in range(n):
        ax = axes[k]
        ax.fill_between(r, 0.0, h[k], color=C_FILL, lw=0, zorder=1)
        ax.plot(r, h[k], color=C_LINE, lw=1.5, zorder=2)
        ax.axvline(R0, color="0.6", lw=0.8, ls=":", zorder=0)
        ax.set_title(r"$t=%g$ s" % t[k], fontsize=11, pad=2)
        ax.grid(alpha=0.22, lw=0.5)
        ax.tick_params(labelsize=9, pad=0.5)
        ax.set_xlim(0.0, HALF)
        ax.set_ylim(0.0, ymax)

    for j in range(ncol):
        axes[(nrow - 1) * ncol + j].set_xlabel(r"$r$ [m]", fontsize=10)
    for i in range(nrow):
        axes[i * ncol].set_ylabel(r"$h$ [m]", fontsize=10)

    fig.suptitle(r"reference radial depth $h(r,t)$ — one panel per output instant"
                 r"   (dotted line: initial dam radius $r_0=3$ m)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    p = os.path.join(outdir, "fig_e2_ref_snap_grid.png")
    fig.savefig(p, dpi=200)
    plt.close(fig)
    print("  saved", p)


def fig_front(D, outdir):
    """激波锋 r_f(t) 单图（宽扁，适合放在快照网格下方）。"""
    t, front = np.asarray(D["t"], float), np.asarray(D["front"], float)
    sel = t >= 0.1 - 1e-9
    coef = np.polyfit(t[sel], front[sel], 1)
    vfit = float(coef[0])

    fig, ax = plt.subplots(figsize=(11.0, 3.0), constrained_layout=True)
    ax.plot(t, front, "o-", color=C_LINE, lw=1.6, ms=5,
            label=r"reference $r_f(t)$")
    ax.plot(t[sel], np.polyval(coef, t[sel]), "--", color="0.3", lw=1.2,
            label=r"fit: %.3f m s$^{-1}$" % vfit)
    ax.axhline(R0, color="0.6", lw=0.9, ls=":")
    ax.text(t[-1], R0, r"  $r_0=3$ m", va="bottom", ha="right",
            fontsize=8, color="0.4")
    ax.set_xlabel(r"$t$ [s]")
    ax.set_ylabel(r"front radius $r_f$ [m]")
    ax.set_title(r"reference front $r_f(t)$   (planar exact %.3f m s$^{-1}$; "
                 r"radial slowdown %.1f%%)"
                 % (S_PLANAR, 100.0 * (1.0 - vfit / S_PLANAR)), fontsize=11)
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(loc="lower right", fontsize=9, frameon=False)

    p = os.path.join(outdir, "fig_e2_ref_front.png")
    fig.savefig(p, dpi=200)
    plt.close(fig)
    print("  saved", p)
    print("  front fit speed = %.4f m/s  (planar exact %.4f, ratio %.4f)"
          % (vfit, S_PLANAR, vfit / S_PLANAR))


def fig_timesections(D, outdir):
    """(a) 径向水深剖面 11 帧叠加；(b) 激波锋位置 r_f(t)。"""
    r = np.asarray(D["r"], float)
    t = np.asarray(D["t"], float)
    h = np.asarray(D["h"], float)
    front = np.asarray(D["front"], float)

    fig, ax = plt.subplots(1, 2, figsize=(11.0, 4.2), constrained_layout=True)

    norm = Normalize(vmin=float(t[0]), vmax=float(t[-1]))
    cmap = cm.viridis
    for k in range(len(t)):
        ax[0].plot(r, h[k], color=cmap(norm(t[k])), lw=1.25)
    ax[0].axvline(R0, color="0.45", ls="--", lw=1.0)
    ax[0].text(R0, ax[0].get_ylim()[1] * 0.97, r" gate $r_0=3$ m",
               ha="left", va="top", fontsize=7.5, color="0.35")
    ax[0].set_xlabel(r"$r$ [m]")
    ax[0].set_ylabel(r"reference depth $h(r,t)$ [m]")
    ax[0].set_title(r"(a) reference radial sections $h(r,t)$")
    ax[0].set_xlim(0.0, HALF)
    ax[0].grid(alpha=0.25, lw=0.5)
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax[0], pad=0.015)
    cb.set_label(r"$t$ [s]")

    sel = t >= 0.1 - 1e-9
    coef = np.polyfit(t[sel], front[sel], 1)
    vfit = float(coef[0])
    ax[1].plot(t, front, "o-", color="#1f77b4", lw=1.5, ms=4,
               label=r"reference $r_f(t)$")
    ax[1].plot(t[sel], np.polyval(coef, t[sel]), "--", color="0.3", lw=1.2,
               label=r"fit: %.3f m s$^{-1}$" % vfit)
    ax[1].axhline(R0, color="0.6", lw=0.9, ls=":")
    ax[1].set_xlabel(r"$t$ [s]")
    ax[1].set_ylabel(r"front radius $r_f$ [m]")
    ax[1].set_title(r"(b) reference front $r_f(t)$  (planar exact %.3f m s$^{-1}$)"
                    % S_PLANAR)
    ax[1].grid(alpha=0.25, lw=0.5)
    ax[1].legend(loc="lower right", fontsize=8, frameon=False)

    p = os.path.join(outdir, "fig_e2_ref_timesections.png")
    fig.savefig(p, dpi=200)
    plt.close(fig)
    print("  saved", p)
    print("  front fit speed = %.4f m/s  (planar exact %.4f, ratio %.4f)"
          % (vfit, S_PLANAR, vfit / S_PLANAR))


def _zeta_on_square(xs, ys, r, h_k):
    """把径向剖面 h(r) 映射到方域 [-10,10]^2：r>10 处用 h(r_max)（静水 1 m）。"""
    Xg, Yg = np.meshgrid(xs, ys, indexing="ij")
    R = np.sqrt((Xg - 0.0) ** 2 + (Yg - 0.0) ** 2)
    Z = np.interp(R, r, h_k)            # np.interp 对 R > r[-1] 返回 h[-1]
    return Xg, Yg, Z


def fig_surface3d(D, outdir, step3d=2):
    """ζ(x¹,x²) 的 3D 快照总览（与 PINN surface3d_evolution 同布局）。

    `step3d` 控制 3D 面板的时间抽样（默认 2 ⇒ t = 0, 0.2, …, 1.0 共 6 帧）；
    时间截面图（fig_timesections）仍画**全部**帧，不受影响。
    """
    r = np.asarray(D["r"], float)
    t = np.asarray(D["t"], float)
    h = np.asarray(D["h"], float)

    keep = list(range(0, len(t), max(1, int(step3d))))
    if keep[-1] != len(t) - 1:
        keep.append(len(t) - 1)

    xs = np.linspace(-HALF, HALF, 121)
    ys = np.linspace(-HALF, HALF, 121)
    nt = len(keep)
    h_lo = float(np.min(h))
    h_hi = float(np.max(h))
    if h_hi - h_lo < 1e-12:
        h_hi = h_lo + 1.0
    cmap = plt.get_cmap("Blues")
    norm = Normalize(vmin=h_lo, vmax=h_hi)

    ncol = min(3, nt)
    nrow = int(np.ceil(nt / ncol))
    fig = plt.figure(figsize=(3.6 * ncol, 3.3 * nrow))
    th = np.linspace(0.0, 2.0 * np.pi, 200)
    for j, k in enumerate(keep):
        ax = fig.add_subplot(nrow, ncol, j + 1, projection="3d")
        Xg, Yg, Z = _zeta_on_square(xs, ys, r, h[k])
        ax.plot_surface(Xg, Yg, Z, cmap=cmap, norm=norm,
                        rstride=_decim(len(xs), 121), cstride=_decim(len(ys), 121),
                        linewidth=0.0, antialiased=False, shade=False)
        ax.plot(R0 * np.cos(th), R0 * np.sin(th), np.full_like(th, h_lo),
                color="0.55", ls="--", lw=0.8, zorder=0)
        ax.set_title(r"$t=%g$ s" % t[k], fontsize=11, pad=-2)
        ax.set_xlabel(r"$x^1$", fontsize=9, labelpad=-6)
        ax.set_ylabel(r"$x^2$", fontsize=9, labelpad=-6)
        ax.set_zlabel(r"$\zeta$ [m]", fontsize=9, labelpad=7)
        ax.set_zlim(h_lo, h_hi)
        ax.tick_params(labelsize=8, pad=-1)
        ax.view_init(elev=28, azim=-58)
        try:
            ax.set_box_aspect((1.0, 1.0, 0.62))
        except Exception:
            pass
    fig.subplots_adjust(right=0.90)
    cax = fig.add_axes([0.915, 0.15, 0.013, 0.70])
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                 label=r"$\zeta$ [m]")
    fig.suptitle(r"Circular dam break, flat bed: reference free surface "
                 r"$\zeta(x^1,x^2,t)$  ($\zeta\equiv h$, $\gamma\equiv 1$)",
                 fontsize=13)
    p = os.path.join(outdir, "fig_e2_ref_surface3d.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  saved", p, " (%d 帧)" % nt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--outdir", default="../figures")
    ap.add_argument("--step3d", type=int, default=2,
                    help="3D 面板的时间抽样步长（默认 2）")
    ap.add_argument("--only-snaps", action="store_true",
                    help="只出快照网格与前锋单图")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    D = load(a.npz)
    print("npz keys:", sorted(D.keys()))
    print("  r in [%.4f, %.4f]  t = %s" % (float(D["r"][0]), float(D["r"][-1]),
                                           np.array2string(np.asarray(D["t"]), precision=2)))
    print("  h(r_max) at t=0 : %.6f   at t=T : %.6f"
          % (float(np.asarray(D["h"])[0, -1]), float(np.asarray(D["h"])[-1, -1])))
    if not a.only_snaps:
        fig_timesections(D, a.outdir)
        fig_surface3d(D, a.outdir, step3d=a.step3d)
    fig_snap_grid(D, a.outdir)
    fig_front(D, a.outdir)


if __name__ == "__main__":
    main()
