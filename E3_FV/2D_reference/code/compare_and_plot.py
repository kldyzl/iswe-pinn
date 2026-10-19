#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
compare_and_plot.py -- same-metric comparison between the trained ISWE-PINN
solution and the high-resolution reference solution, for the two dam-break
experiments of the paper.

  E2 : circular dam break on the flat bed            (reference: swe_ref.py, radial)
  E3 : dam break on the parabolic bed (Bachini2020)   (reference: swe_ref.py, intrinsic)

Writes into --outdir:
    compare_<TAG>.txt               the numbers (profiles, front, scalars, budgets)
and into --figdir (default = --outdir):
    fig_<tag>_refsol_compare.png    the overlay figure used in the paper

Nothing is recomputed here: both fields are read from the run artefacts.

Usage
-----
  python compare_and_plot.py --case e2 \
      --ref ../output/e2_n8000.npz \
      --pinn ../../pinn/output/pinn_dam_circ_flat.npz \
      --outdir ../output  --figdir ../figures --tag E2

  python compare_and_plot.py --case e3 \
      --ref ../output/e3_n8000.npz \
      --pinn ../../pinn/output/pinn_dam_parab_p1_snap.npz \
      --outdir ../output  --figdir ../figures --tag E3
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EPS = 1e-12
_trapz = getattr(np, "trapezoid", None) or np.trapz

# exact planar dam-break shock speed for hL = 2, hR = 1, u = 0  (swe_ref.py)
S_PLANAR = 4.183127921958328
R0_DAM = 3.0
SQ_CORNER = 400.0 - 100.0 * np.pi          # [-10,10]^2 minus the disc r < 10

# --- figure style (print, greyscale-safe line styles) -----------------------
plt.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
})


def _load(path):
    if path is None or not os.path.exists(path):
        return None
    return np.load(path, allow_pickle=True)


def _front(x, f):
    """argmax of |df/dx| away from the two end points."""
    d = np.abs(np.gradient(f, x))
    k = int(np.argmax(d[1:-1])) + 1
    return float(x[k])


# ---------------------------------------------------------------------------
#  E2 -- circular dam break on the flat bed
# ---------------------------------------------------------------------------
def run_e2(a):
    ref = _load(a.ref)
    pin = _load(a.pinn)
    if ref is None or pin is None:
        return None
    out = []
    W = out.append

    r_ref, t_ref, h_ref = (np.asarray(ref[k], float)
                           for k in ("r", "t", "h"))
    t_pin, r_pin, h_pin = (np.asarray(pin[k], float)
                           for k in ("t_list", "r_prof", "h_prof"))
    x_half = float(np.asarray(pin["x"], float)[-1])
    L_dom = 2.0 * x_half

    W("=" * 78)
    W("E2  circular dam break, h(r) on the flat bed      domain length %.1f m" % L_dom)
    W("=" * 78)
    W("  %6s %12s %12s %12s %12s %10s" %
      ("t [s]", "rel L1", "Linf [m]", "front PINN", "front ref", "|dx|/L"))
    rows = []
    for kp, tp in enumerate(t_pin):
        kr = int(np.argmin(np.abs(t_ref - tp)))
        if abs(t_ref[kr] - tp) > 1e-9:
            continue
        rp = np.asarray(r_pin[kp], float)
        hp = np.asarray(h_pin[kp], float)
        href = np.interp(rp, r_ref, h_ref[kr])
        rel = float(np.sum(np.abs(hp - href)) / max(np.sum(np.abs(href)), EPS))
        lin = float(np.max(np.abs(hp - href)))
        xfp, xfr = _front(rp, hp), _front(r_ref, h_ref[kr])
        W("  %6.2f %12.4e %12.4e %12.4f %12.4f %10.2e"
          % (tp, rel, lin, xfp, xfr, abs(xfp - xfr) / L_dom))
        rows.append(dict(t=float(tp), rel=rel, lin=lin, fp=xfp, fr=xfr,
                         fe=abs(xfp - xfr) / L_dom))
    W("  %-14s rel L1: mean %.4e  max %.4e" %
      ("summary", np.mean([q["rel"] for q in rows]), np.max([q["rel"] for q in rows])))
    W("  %-14s Linf:   max %.4e m" % ("", np.max([q["lin"] for q in rows])))
    W("  %-14s front error / L: max %.4e" % ("", np.max([q["fe"] for q in rows])))

    # front speed, fitted on 0.1 s <= t <= 1 s (the initial step is smeared)
    sel = [q for q in rows if q["t"] >= 0.1 - 1e-9]
    tt = np.array([q["t"] for q in sel])
    vp = float(np.polyfit(tt, [q["fp"] for q in sel], 1)[0])
    vr = float(np.polyfit(tt, [q["fr"] for q in sel], 1)[0])
    W("")
    W("  fitted front speed, 0.1 s <= t <= 1 s:")
    W("      PINN      %.4f m/s" % vp)
    W("      reference %.4f m/s     (planar exact %.4f m/s  ->  ratio %.4f)"
      % (vr, S_PLANAR, vr / S_PLANAR))

    # mass: the PINN integrates over the square, the reference over the radius
    m_ref = np.asarray(ref["mass"], float)
    m_pin = np.asarray(pin["mass"], float)
    W("")
    W("  mass, PINN square measure vs equivalent reference measure")
    W("      M_square = 2*pi*M_radial + %.3f   (corner region at h = 1 m)" % SQ_CORNER)
    W("  %6s %16s %16s %12s" % ("t [s]", "M PINN (square)", "M ref equiv.", "rel diff"))
    eq = 2.0 * np.pi * m_ref + SQ_CORNER
    for kp, tp in enumerate(t_pin):
        kr = int(np.argmin(np.abs(t_ref - tp)))
        W("  %6.2f %16.6f %16.6f %12.2e"
          % (tp, m_pin[kp], eq[kr], abs(m_pin[kp] - eq[kr]) / eq[kr]))
    rdm = [abs(m_pin[k] - eq[int(np.argmin(np.abs(t_ref - t)))])
           / eq[int(np.argmin(np.abs(t_ref - t)))] for k, t in enumerate(t_pin)]
    W("  %-14s max %.4e" % ("summary", np.max(rdm)))
    W("")
    W("  continuum invariant of the reference: max|M(t)-M(0)|/M(0) = %.3e"
      % (np.max(np.abs(m_ref - m_ref[0])) / m_ref[0]))
    W("  reference M(0) in the radial measure = %.6f" % m_ref[0])

    # ------------------------------------------------------------------ figure
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 3.1))
    show = [0.1, 0.3, 0.5, 0.7, 1.0]
    cmap = plt.get_cmap("viridis")
    pairs = []
    for i, ts in enumerate(show):
        c = cmap(i / (len(show) - 1.0))
        kr = int(np.argmin(np.abs(t_ref - ts)))
        kp = int(np.argmin(np.abs(t_pin - ts)))
        lr, = ax[0].plot(r_ref, h_ref[kr], "-", color=c, lw=1.3)
        lp, = ax[0].plot(r_pin[kp], h_pin[kp], "--", color=c, lw=1.3)
        pairs.append((lr, lp))
    ax[0].axvline(R0_DAM, color="0.4", lw=0.9, ls=":")
    ax[0].set_xlabel(r"$r$ [m]")
    ax[0].set_ylabel(r"$h$ [m]")
    ax[0].set_title("(a) radial profiles   (solid: reference, dashed: PINN)")
    ax[0].set_xlim(0, 8.2)
    ax[0].legend([tuple(p) for p in pairs], ["$%.1f$" % t for t in show],
                 ncol=len(show), loc="upper center", bbox_to_anchor=(0.5, -0.17),
                 frameon=False, fontsize=6.8, handlelength=2.6,
                 columnspacing=1.1, handletextpad=0.5)

    ax[1].plot(t_ref, ref["front"], "o-", color="#c0392b", ms=3.5, lw=1.3,
               label="reference")
    ax[1].plot(t_pin, pin["shock_r"], "s--", color="#1f4e79", ms=3.5, lw=1.3,
               label="PINN")
    ax[1].plot(t_ref, R0_DAM + S_PLANAR * t_ref, "-", color="0.35", lw=1.0,
               label="planar exact (%.3f m/s)" % S_PLANAR)
    ax[1].set_xlabel(r"$t$ [s]")
    ax[1].set_ylabel(r"front radius $r_{\rm f}$ [m]")
    ax[1].set_title("(b) front position")
    ax[1].legend(loc="upper left", framealpha=0.9)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.90, bottom=0.235,
                        wspace=0.28)
    fp = os.path.join(a.figdir, "fig_%s_refsol_compare.png" % a.tag.lower())
    fig.savefig(fp)
    plt.close(fig)
    W("")
    W("  figure: %s" % os.path.basename(fp))
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
#  E3 -- dam break on the parabolic bed
# ---------------------------------------------------------------------------
def run_e3(a):
    ref = _load(a.ref)
    pin = _load(a.pinn)
    if ref is None or pin is None:
        return None
    out = []
    W = out.append

    x_ref, t_ref, eta_ref = (np.asarray(ref[k], float)
                             for k in ("x", "t", "eta"))
    t_pin = np.asarray(pin["t_snap"], float)
    x_pin = np.asarray(pin["prof_x"], float)
    h_pin = np.asarray(pin["prof_h_snap"], float)
    L_dom = float(x_pin[-1] - x_pin[0])

    W("=" * 78)
    W("E3  parabolic-bed dam break, h(x) mid-channel      domain length %.1f m" % L_dom)
    W("=" * 78)
    W("  %6s %12s %12s" % ("t [s]", "rel L1", "Linf [m]"))
    rows = []
    for kp, tp in enumerate(t_pin):
        kr = int(np.argmin(np.abs(t_ref - tp)))
        if abs(t_ref[kr] - tp) > 1e-9:
            continue
        eref = np.interp(x_pin, x_ref, eta_ref[kr])
        rel = float(np.sum(np.abs(h_pin[kp] - eref)) / max(np.sum(np.abs(eref)), EPS))
        lin = float(np.max(np.abs(h_pin[kp] - eref)))
        W("  %6.2f %12.4e %12.4e" % (tp, rel, lin))
        rows.append((rel, lin))
    W("  %-14s rel L1: mean %.4e  max %.4e" %
      ("summary", np.mean([q[0] for q in rows]), np.max([q[0] for q in rows])))
    W("  %-14s Linf:   max %.4e m" % ("", np.max([q[1] for q in rows])))

    # ---- unambiguous scalars: outlet discharge and mass -------------------
    q_ref = np.asarray(ref["q_out"], float)
    q_pin = np.asarray(pin["q_out_snap"], float)
    m_ref = np.asarray(ref["mass"], float)
    m_pin = np.asarray(pin["mass_snap"], float)
    W("")
    W("  outlet discharge and mass (no reference field is interpolated here)")
    W("  %6s %12s %12s %8s %14s %14s" %
      ("t [s]", "Q_out PINN", "Q_out ref", "ratio", "M(t) PINN", "M(t) ref"))
    for kp, tp in enumerate(t_pin):
        kr = int(np.argmin(np.abs(t_ref - tp)))
        rr = q_pin[kp] / q_ref[kr] if q_ref[kr] > EPS else float("nan")
        W("  %6.2f %12.5f %12.5f %8.3f %14.6f %14.6f"
          % (tp, q_pin[kp], q_ref[kr], rr, m_pin[kp], m_ref[kr]))

    dM_p = m_pin[0] - m_pin[-1]
    Iq_p = float(_trapz(q_pin, t_pin))
    Iq_r = float(ref["int_q_out"][-1])
    dM_r = m_ref[0] - m_ref[-1]
    W("")
    W("  mass budget over [0, T]:")
    W("      PINN      dM = %.6f   int Q_out dt = %.6f   dM/int Q = %.3f"
      % (dM_p, Iq_p, dM_p / max(Iq_p, EPS)))
    W("      reference dM = %.6f   int Q_out dt = %.6f   dM/int Q = %.3f"
      % (dM_r, Iq_r, dM_r / max(Iq_r, EPS)))
    W("      (the reference closes its own budget identically: the cumulative")
    W("       outlet integral above is accumulated with the same RK2 weights as M,")
    W("       so the two numbers agree to round-off by construction.)")
    W("")
    W("  NOTE.  The front detector argmax|dh/dx| is *not* used for this case:")
    W("         from t ~ 1 s the leading edge decelerates on the down-slope and")
    W("         heaps up, so the detector locks onto the heap and not the front.")

    # ------------------------------------------------------------------ figure
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 3.0))
    cmap = plt.get_cmap("viridis")
    show = [0.0, 0.5, 1.0, 1.5]
    pairs = []
    for i, ts in enumerate(show):
        c = cmap(i / (len(show) - 1.0) * 0.85)
        kr = int(np.argmin(np.abs(t_ref - ts)))
        kp = int(np.argmin(np.abs(t_pin - ts)))
        lr, = ax[0].plot(x_ref, eta_ref[kr], "-", color=c, lw=1.3)
        lp, = ax[0].plot(x_pin, h_pin[kp], "--", color=c, lw=1.3)
        pairs.append((lr, lp))
    ax[0].set_xlabel(r"$x^1$ [m]")
    ax[0].set_ylabel(r"$h$ [m]")
    ax[0].set_title("(a) longitudinal profiles")
    ax[0].legend([tuple(p) for p in pairs], ["$%.1f$" % t for t in show],
                 ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.17),
                 frameon=False, fontsize=6.8, handlelength=2.6,
                 columnspacing=0.8, handletextpad=0.4)

    ax[1].plot(t_ref, q_ref, "o-", color="#c0392b", ms=3.5, lw=1.3, label="reference")
    ax[1].plot(t_pin, q_pin, "s--", color="#1f4e79", ms=3.5, lw=1.3, label="PINN")
    ax[1].set_xlabel(r"$t$ [s]")
    ax[1].set_ylabel(r"$Q_{\rm out}$ [m$^3$s$^{-1}$]")
    ax[1].set_title("(b) outlet discharge")
    ax[1].legend(loc="upper left", framealpha=0.9)

    ax[2].plot(t_ref, m_ref, "o-", color="#c0392b", ms=3.5, lw=1.3, label="reference")
    ax[2].plot(t_pin, m_pin, "s--", color="#1f4e79", ms=3.5, lw=1.3, label="PINN")
    ax[2].set_xlabel(r"$t$ [s]")
    ax[2].set_ylabel(r"$M(t)$ [m$^3$]")
    ax[2].set_title("(c) mass")
    ax[2].legend(loc="upper right", framealpha=0.9)
    for aa in ax[1:]:                      # time axes only; (a) is a space axis
        aa.set_xlim(0, 1.5)
    fig.subplots_adjust(left=0.065, right=0.985, top=0.90, bottom=0.235,
                        wspace=0.42)
    fp = os.path.join(a.figdir, "fig_%s_refsol_compare.png" % a.tag.lower())
    fig.savefig(fp)
    plt.close(fig)
    W("")
    W("  figure: %s" % os.path.basename(fp))
    return "\n".join(out) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--case", choices=("e2", "e3"), required=True)
    p.add_argument("--ref", required=True)
    p.add_argument("--pinn", required=True)
    p.add_argument("--outdir", default=".")
    p.add_argument("--figdir", default=None,
                   help="where to put the figure (default: same as --outdir)")
    p.add_argument("--tag", default=None)
    a = p.parse_args()
    if a.tag is None:
        a.tag = a.case.upper()
    if a.figdir is None:
        a.figdir = a.outdir
    os.makedirs(a.outdir, exist_ok=True)
    os.makedirs(a.figdir, exist_ok=True)
    txt = run_e2(a) if a.case == "e2" else run_e3(a)
    if txt is None:
        raise SystemExit("missing input; nothing written")
    fp = os.path.join(a.outdir, "compare_%s.txt" % a.tag)
    with open(fp, "w", encoding="utf-8") as fh:
        fh.write("PINN vs reference solution (swe_ref.py)\n\n" + txt)
    print(txt)
    print("written: %s" % fp)


if __name__ == "__main__":
    main()
