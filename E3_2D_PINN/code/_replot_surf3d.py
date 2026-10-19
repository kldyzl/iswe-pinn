# -*- coding: utf-8 -*-
r"""离线重绘 3D 演化图：复用主脚本的绘图函数，在已保存的 snap.npz 上出图。

用法：
    python _replot_surf3d.py <snap.npz> <outdir>

只做绘图，不训练、不碰 GPU（matplotlib Agg + numpy）。
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import iswe_pinn_e2p_parabola as M       # noqa: E402


def main():
    npz, outdir = sys.argv[1], sys.argv[2]
    os.makedirs(outdir, exist_ok=True)
    d = np.load(npz, allow_pickle=True)
    ev = {}
    for k in d.files:
        v = d[k]
        if v.dtype == object and v.shape == ():
            v = v.item()
        ev[k] = v
    # npz 里存的是 "h_snap" / "zeta_snap" / "t_snap"（来自 evaluate_field 的 ev 字典）
    ren = {"h_snap": "h", "u1_snap": "u1", "u2_snap": "u2",
           "zeta_snap": "zeta", "t_snap": "t_list"}
    for a, b in ren.items():
        if a in ev and b not in ev:
            ev[b] = ev[a]
    print("ev keys:", sorted(ev.keys()))
    print("h.shape", np.asarray(ev["h"]).shape,
          "zeta.shape", np.asarray(ev["zeta"]).shape,
          "nt", len(np.asarray(ev["t_list"])))
    files = M.make_surface_evolution_plot(ev, outdir)
    for f in files:
        print("wrote", f, os.path.getsize(f), "bytes")
    files2 = M.make_surface_single_plots(ev, outdir)
    print("single plots:", len(files2))


if __name__ == "__main__":
    main()
