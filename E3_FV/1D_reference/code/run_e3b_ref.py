#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run_e3b_ref.py — E3b 参考解 **非侵入式驱动**

铁律（计划 R2-1 / skill G2）：
  * 本文件**只 import `swe_ref`**，不复制、不重写任何数学；
  * `swe_ref.py` 本体**逐字节不变**（md5 = efcdb3c63bff8e9460cbd23c812379f6）；
  * 新增功能（批量网格、自收敛、网格收敛门）全部写在本驱动里。

用法
----
  python run_e3b_ref.py --n 16000 --out ../output/e3b_n16000.npz
  python run_e3b_ref.py --selfconv 1000,2000,4000,8000,16000 \\
                        --out-eoc ../output/selfconv_eoc.txt --save-finest ../output/e3b_n16000.npz
  python run_e3b_ref.py --verify
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import swe_ref as R  # noqa: E402  —— 唯一依赖；**只读调用，不修改**

# 网格收敛门（预注册，见 README §6 判据④）
GATE_LO, GATE_HI, GATE_TOL = 8000, 16000, 1e-3


def rel_l1(eta_coarse, xc_coarse, eta_fine, xc_fine):
    """Σ|η_c→f − η_f| / Σ|η_f|（粗网格线性插值到细网格单元中心）。"""
    e_c = np.interp(xc_fine, xc_coarse, eta_coarse)
    den = float(np.sum(np.abs(eta_fine)))
    if den <= 0.0:
        return float("nan")
    return float(np.sum(np.abs(e_c - eta_fine)) / den)


def run_one(n, verbose=True):
    return R.run_e3(n=n, tend=R.T_E3, cfl=0.4, snap_every=0.1, verbose=verbose)


def save_npz(res, path):
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    np.savez_compressed(
        path,
        x=res["x"], xf=res["xf"], t=res["t"], eta=res["eta"], q=res["q"],
        zeta=res["zeta"], mass=res["mass"], front=res["front"],
        mass0=res["mass0"], t_q=res["t_q"], int_q_out=res["int_q_out"],
        m_all=res["m_all"], q_out=res["q_out"], n=res["n"],
        steps=res["steps"], wall=res["wall"], cfl=res["cfl"], tend=res["tend"],
        g=res["g"],
    )
    print("  saved %s  (%.1f KB)" % (path, os.path.getsize(path) / 1e3))


def print_ref_gates(res):
    """判据 ②（自收支）与 ⑤（出口 ∂ζ/∂x 留档）的可执行检查。"""
    mass0 = res["mass0"]
    dM = mass0 - res["mass"][-1]
    iq = res["int_q_out"][-1]
    rel = abs(iq - dM) / abs(dM) if dM != 0.0 else float("nan")
    print("  [判据②] |∫Q_out dt − ΔM|/ΔM = %.3e   (须 ≤ 1e-12)" % rel)
    xc, zeta = res["x"], res["zeta"]
    dx = xc[1] - xc[0]
    # 二阶**后向**差分（正确符号）： f'(x_N) = (3 f_N − 4 f_{N−1} + f_{N−2}) / (2h)
    dz = (3.0 * zeta[:, -1] - 4.0 * zeta[:, -2] + zeta[:, -3]) / (2.0 * dx)
    print("  [判据⑤] 出口 ∂ζ/∂x 逐时刻（二阶后向差分）：")
    for k, t in enumerate(res["t"]):
        print("            t=%.1f  ∂ζ/∂x(10)=%+.6f" % (t, dz[k]))
    return dict(rel=rel, dz=dz)


def cmd_run(a):
    res = run_one(a.n)
    save_npz(res, a.out)
    print_ref_gates(res)


def cmd_selfconv(a):
    ns = [int(s) for s in a.selfconv.split(",")]
    print("=== 自收敛：n = %s ===" % ns)
    res = {}
    for n in ns:
        print("-- n=%d --" % n)
        res[n] = run_one(n)
    rows = []
    for i in range(len(ns) - 1):
        nc, nf = ns[i], ns[i + 1]
        rc, rf = res[nc], res[nf]
        vals = [rel_l1(rc["eta"][k], rc["x"], rf["eta"][k], rf["x"])
                for k in range(len(rf["t"]))]
        rows.append((nc, nf, max(vals)))

    lines = [
        "E3b reference self-convergence (cell-centred eta; linear interp to fine grid)",
        "rel-L1(a,b) = sum|eta_a->b - eta_b| / sum|eta_b|,  max over t in [0,1.5]",
        "",
        "    pair(a,b)        rel-L1_max          EOC",
    ]
    for i, (nc, nf, v) in enumerate(rows):
        eoc = ""
        if i > 0:
            prev = rows[i - 1][2]
            if v > 0.0 and prev > 0.0:
                eoc = "%.3f" % (np.log(prev / v) / np.log(float(nf) / nc))
        lines.append("  (%5d,%5d)   %.6e     %s" % (nc, nf, v, eoc))
    gate = next((v for nc, nf, v in rows if (nc, nf) == (GATE_LO, GATE_HI)), None)
    lines.append("")
    if gate is None:
        lines.append("GATE: n=(%d,%d) 未在本次网格序列中 -> 无法判定" % (GATE_LO, GATE_HI))
    else:
        lines.append("GATE rel-L1(%d,%d) = %.6e   (须 < %.0e)  -> %s"
                     % (GATE_LO, GATE_HI, gate, GATE_TOL,
                        "PASS" if gate < GATE_TOL else "FAIL"))
    txt = "\n".join(lines)
    print(txt)
    if a.out_eoc:
        d = os.path.dirname(os.path.abspath(a.out_eoc))
        if d:
            os.makedirs(d, exist_ok=True)
        with open(a.out_eoc, "w", encoding="utf-8") as f:
            f.write(txt + "\n")
        print("  saved %s" % a.out_eoc)
    if a.save_finest:
        save_npz(res[ns[-1]], a.save_finest)


def cmd_verify(a):
    R.main(["--verify"])


def parse_args():
    p = argparse.ArgumentParser(description="E3b reference driver (non-invasive)")
    p.add_argument("--n", type=int, default=16000)
    p.add_argument("--out", default=None)
    p.add_argument("--selfconv", default=None, help="逗号分隔的网格序列，如 1000,2000,4000,8000,16000")
    p.add_argument("--out-eoc", default=None)
    p.add_argument("--save-finest", default=None)
    p.add_argument("--verify", action="store_true")
    return p.parse_args()


def main():
    a = parse_args()
    if a.verify:
        cmd_verify(a)
    elif a.selfconv:
        cmd_selfconv(a)
    elif a.out:
        cmd_run(a)
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
