#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
swe_ref.py --- one-dimensional high-resolution reference solutions for the two
dam-break experiments of the ISWE-PINN study.

WHY THIS FILE EXISTS
--------------------
Section 4 of the manuscript measures the network error against a reference
solution.  The references declared there are produced here, on grids fine
enough that the discretisation error is at least an order of magnitude below
the network error.

  E2  flat bed, circular dam break, domain [-10,10]^2, solid walls, T = 1 s,
      h = 2 for r <= 3 and h = 1 otherwise, u = 0.
      Data and geometry are radially symmetric and the outward front stays
      inside r = 7 (measured below), so the restriction of the radially
      symmetric solution to the square satisfies the square-domain
      initial-boundary value problem and is *the* solution there.  The 2D
      problem therefore reduces exactly to the 1D SWE with varying width
      b(r) = r --- the classical reduction of the circular dam break --- and
      the finite volume fluxes are weighted by the face radius, so that mass
      is conserved to machine precision (verified below).

  E3  parabolic bed z = (x-10)^2/25 on [0,10] x [0,1], dam at x = 2,
      eta = 0.5 (x < 2) and 0.2 (x > 2), q = 0, wall at x = 0, free outflow at
      x = 10, T = 1.5 s.  The solution is x2-independent, so the ISWE reduce to
      a 1D intrinsic system in s^1 = x, discretised with the first-order
      intrinsic Godunov finite volume scheme of Bachini & Putti (2020) --- the
      scheme the manuscript cites as its reference discretisation: HLL flux on
      the frozen edge metric, hydrostatic reconstruction of the normal depth
      from the geopotential z + gamma*eta, and the geometric source folded into
      the edge flux so that the lake at rest is preserved exactly.

WHERE THE SCHEME COMES FROM (verified sources; links in README.md)
------------------------------------------------------------------
 [BP20]  E. Bachini, M. Putti, "Geometrically intrinsic modeling of shallow
         water flows", ESAIM: M2AN 54(6) 2125-2157 (2020),
         doi:10.1051/m2an/2020031 --- sections 2.3 (ISWE), 3.1-3.3 (intrinsic
         Godunov FV scheme: (3.4) cell/edge averages, (3.5) update with the
         source folded into the flux, (3.11) Riemann problem, (3.12) parallel
         transport, section 3.2.1 HLL with wave speeds v +- sqrt(g eta gamma),
         (3.14)-(3.16) well-balanced source quadrature), 4.2 (the parabola test
         case, whose parameters are used verbatim for E3).
 [A04]   E. Audusse, F. Bouchut, M.-O. Bristeau, R. Klein, B. Perthame, SIAM J.
         Sci. Comput. 25(6) 2050-2065 (2004), doi:10.1137/S1064827503431090 ---
         hydrostatic reconstruction, face bed elevation max{z_L,z_R,z(x_f)},
         positivity.
 [HLL83] A. Harten, P. D. Lax, B. van Leer, SIAM Review 25(1) 35-61 (1983) ---
         the HLL flux.
 [T01]   E. F. Toro, "Shock-Capturing Methods for Free-Surface Shallow Flows",
         Wiley (2001) --- exact dam-break Riemann solution used in the audit.
 [MK14]  S. Mungkasi, J. Phys. Conf. Ser. 539 012022 (2014),
         doi:10.1088/1742-6596/539/1/012022 --- the circular dam break as the 1D
         SWE with varying width b(x) (their eqs. (1)-(2)); the circular shock is
         slower than the planar one.
 [ZR00]  C. Zoppou, S. Roberts, Appl. Math. Modelling 24(7) 457-475 (2000),
         doi:10.1016/S0307-904X(99)00056-6 --- the same radial reduction (their
         eqs. (19)-(20)); Cartesian-grid artefacts of a circular dam.
 [DW13]  O. Delestre et al., Int. J. Numer. Meth. Fluids 72(3) 269-300 (2013),
         doi:10.1002/fld.3741, arXiv:1110.0288 --- analytic SWE benchmarks.

Usage
-----
    python3 swe_ref.py --verify                  # audit battery, writes nothing
    python3 swe_ref.py --verify --with-2d        # + Cartesian 2D cross-check
    python3 swe_ref.py --case e2 --n 16000       # writes out/e2_n16000.npz
    python3 swe_ref.py --case e3 --n 16000
"""

import argparse
import os
import sys
import time

import numpy as np

# numpy 2.0 renamed trapz -> trapezoid and dropped the old name; support both so
# that the script runs identically on the local interpreter and in venv.
_trapz = getattr(np, "trapezoid", None)
if _trapz is None:                      # numpy < 2
    _trapz = np.trapz

G = 9.81

R_OUT_E2 = 10.0
DAM_R = 3.0
H_IN, H_OUT = 2.0, 1.0
T_E2 = 1.0

L_E3 = 10.0
DAM_X = 2.0
ETA_UP, ETA_DN = 0.5, 0.2
T_E3 = 1.5
Z_COEF = 1.0 / 25.0

EPS = 1e-12

# Dry-state threshold.  A cell or an edge whose depth is at or below H_DRY m
# carries no discharge: this is the non-negativity closure that [BP20] invoke
# through their bound eta* = max{0, ...} at the edge reconstruction, made
# computable (see e3_faces and run_e3).  Without it a drying cell keeps a finite
# q while eta -> 0, the edge velocity v = q/(gamma eta) diverges, the HLL wave
# speeds blow up and the run overflows -- exactly the failure the E3
# parabolic-bed dam break produces at the closed upstream wall, where the bed is
# highest (z(0) = 4 m) and the reservoir end therefore dries.  H_DRY is three
# orders of magnitude below the smallest physical depth of interest (0.2 m), so
# it never touches the wet solution: audit 6.3 (lake at rest), 6.4 (consistency
# order) and 6.6 (self-convergence) are unaffected, and 6.5 shows the mass
# budget is still closed to roundoff because the guard removes no mass -- it
# only zeroes the discharge.
H_DRY = 1e-6


# ============================================================================
# 1. Geometry of the parabolic benchmark bed (E3); closed form everywhere
# ============================================================================
def z3(x):
    """Bed elevation, z = (x-10)^2/25.                              [BP20] (4.1)"""
    return Z_COEF * (x - L_E3) ** 2


def z3p(x):
    """dz/dx."""
    return 2.0 * Z_COEF * (x - L_E3)


def gam3(x):
    """gamma = dx^3/ds^3 = 1/sqrt(1+|grad z|^2).                    [BP20] (2.5)"""
    zp = z3p(x)
    return 1.0 / np.sqrt(1.0 + zp * zp)


def A3(x):
    """h_(1) = |t_1| = 1/gamma;  J = h_(1) h_(2) = A3 for this bed."""
    return 1.0 / gam3(x)


def A3p(x):
    """dA/dx = z' z''/sqrt(1+z'^2)."""
    return z3p(x) * (2.0 * Z_COEF) / np.sqrt(1.0 + z3p(x) ** 2)


def gam3p(x):
    """dgamma/dx = -gamma^3 z' z''."""
    return -(gam3(x) ** 3) * z3p(x) * (2.0 * Z_COEF)


def christoffel3(x):
    """Gamma^1_11 = d_x h_(1)/h_(1) = A'/A (only non-vanishing symbol)."""
    return A3p(x) / A3(x)


# ============================================================================
# 2. HLL Riemann solvers (vectorised over faces)
# ============================================================================
def _u_of(h, m):
    return np.where(h > EPS, m / np.maximum(h, EPS), 0.0)


def hll_swe(hL, mL, hR, mR, g=G):
    r"""
    HLL flux of the classical 1D SWE in U = (h, m = h u), F = (m, m^2/h + g h^2/2),
    with the Davis wave speeds
        S_L = min(uL-cL, uR-cR),  S_R = max(uL+cL, uR+cR),  c = sqrt(g h).
    [HLL83]; [T01] ch.10; cf. clawpack shallow_1D_py.shallow_hll_1D.
    """
    uL, uR = _u_of(hL, mL), _u_of(hR, mR)
    cL = np.sqrt(g * np.maximum(hL, 0.0))
    cR = np.sqrt(g * np.maximum(hR, 0.0))
    SL = np.minimum(uL - cL, uR - cR)
    SR = np.maximum(uL + cL, uR + cR)
    FLh, FLm = mL, mL * uL + 0.5 * g * hL * hL
    FRh, FRm = mR, mR * uR + 0.5 * g * hR * hR
    den = np.where(SR - SL > EPS, SR - SL, 1.0)
    mid_h = (SR * FLh - SL * FRh + SL * SR * (hR - hL)) / den
    mid_m = (SR * FLm - SL * FRm + SL * SR * (mR - mL)) / den
    Fh = np.where(SL >= 0.0, FLh, np.where(SR <= 0.0, FRh, mid_h))
    Fm = np.where(SL >= 0.0, FLm, np.where(SR <= 0.0, FRm, mid_m))
    return Fh, Fm


def hll_swe_2d(h, mx, my, nx, ny, g=G):
    r"""
    HLL flux of the full 2D shallow-water system across a face with unit normal
    (nx, ny).  Unlike the dimension-by-dimension form, all three components of
    the flux are produced by one Riemann solve, so the cross momentum fluxes
    m_x m_y / h are represented correctly -- omitting them lets a dimensionally
    split scheme create energy on a 2D dam break and the solution grows without
    bound.  States U = (h, m_x, m_y), normal velocity u_n = (m_x nx + m_y ny)/h,

        F_d = ( m_n ,  m_x u_n + (g/2) h^2 n_x ,  m_y u_n + (g/2) h^2 n_y ),

    with the Davis wave speeds S_L = min(u_nL - cL, u_nR - cR), S_R = max(...),
    c = sqrt(g h).  [Toro 2001, ch. 10-11]; same HLL construction as hll_swe.
    """
    hL, hR = np.maximum(h[..., 0], 0.0), np.maximum(h[..., 1], 0.0)
    mL, mR = mx[..., 0], mx[..., 1]
    nL, nR = my[..., 0], my[..., 1]
    uL = np.where(hL > EPS, mL / np.maximum(hL, EPS), 0.0)
    vL = np.where(hL > EPS, nL / np.maximum(hL, EPS), 0.0)
    uR = np.where(hR > EPS, mR / np.maximum(hR, EPS), 0.0)
    vR = np.where(hR > EPS, nR / np.maximum(hR, EPS), 0.0)
    uLn, uRn = uL * nx + vL * ny, uR * nx + vR * ny
    cL, cR = np.sqrt(g * hL), np.sqrt(g * hR)
    SL = np.minimum(uLn - cL, uRn - cR)
    SR = np.maximum(uLn + cL, uRn + cR)
    den = np.where(SR - SL > EPS, SR - SL, 1.0)

    def comp(FL, FR, UL, UR):
        mid = (SR * FL - SL * FR + SL * SR * (UR - UL)) / den
        return np.where(SL >= 0.0, FL, np.where(SR <= 0.0, FR, mid))

    pL, pR = 0.5 * g * hL * hL, 0.5 * g * hR * hR
    Fh = comp(mL * nx + nL * ny, mR * nx + nR * ny, hL, hR)
    Fmx = comp(mL * uLn + pL * nx, mR * uRn + pR * nx, mL, mR)
    Fmy = comp(nL * uLn + pL * ny, nR * uRn + pR * ny, nL, nR)
    return Fh, Fmx, Fmy


def hll_intrinsic(etaL, qL, etaR, qR, g, gam_f, A_f):
    r"""
    HLL flux of the frozen-metric intrinsic Riemann problem at a face, in the
    contravariant variables U = (eta, q^1), with the edge geometry held fixed:

        F(eta, q) = ( A_f q ,  A_f q^2/eta + g gam_f^2 eta^2/2 ),

    obtained from the surface-measure flux  Phi = A F^{11}  with
    F^{11} = q^2/eta + g eta^2 gamma^3/2 and A gamma = 1, i.e.
    Phi = A_f q^2/eta + g gam_f^2 eta^2/2  (the A_f factor multiplies only the
    kinetic term).  Eigenvalues v +- sqrt(g eta gam_f), v = A_f q/eta: the wave
    speeds stated in [BP20] section 3.2.1, re-derived from the analytic Jacobian
    in audit 6.4.
    """
    vL = np.where(etaL > EPS, A_f * qL / np.maximum(etaL, EPS), 0.0)
    vR = np.where(etaR > EPS, A_f * qR / np.maximum(etaR, EPS), 0.0)
    cL = np.sqrt(g * np.maximum(etaL, 0.0) * gam_f)
    cR = np.sqrt(g * np.maximum(etaR, 0.0) * gam_f)
    SL = np.minimum(np.minimum(vL - cL, vR - cR), 0.0)
    SR = np.maximum(np.maximum(vL + cL, vR + cR), 0.0)
    FLh = A_f * qL
    FRh = A_f * qR
    # Phi = A_f q^2/eta + (g/2) gam_f^2 eta^2 : the metric factor multiplies the
    # kinetic term only, so that at rest Phi equals (g/2) gam_f^2 (eta*)^2 and
    # cancels the folded source below identically.
    FLm = np.where(etaL > EPS,
                   A_f * qL * qL / np.maximum(etaL, EPS)
                   + 0.5 * g * gam_f ** 2 * etaL ** 2, 0.0)
    FRm = np.where(etaR > EPS,
                   A_f * qR * qR / np.maximum(etaR, EPS)
                   + 0.5 * g * gam_f ** 2 * etaR ** 2, 0.0)
    den = np.where(SR - SL > EPS, SR - SL, 1.0)
    mid_h = (SR * FLh - SL * FRh + SL * SR * (etaR - etaL)) / den
    mid_m = (SR * FLm - SL * FRm + SL * SR * (qR - qL)) / den
    Fh = np.where(SL >= 0.0, FLh, np.where(SR <= 0.0, FRh, mid_h))
    Fm = np.where(SL >= 0.0, FLm, np.where(SR <= 0.0, FRm, mid_m))
    return Fh, Fm


# ============================================================================
# 3. E2: circular dam break as 1D SWE with varying width b(r) = r
# ============================================================================
def grid_e2(n):
    rf = np.linspace(0.0, R_OUT_E2, n + 1)
    rc = 0.5 * (rf[:-1] + rf[1:])
    return rc, rf


def u0_e2(rc):
    return np.where(rc < DAM_R, H_IN, H_OUT), np.zeros_like(rc)


def rhs_e2(h, m, rc, rf, dr, g=G):
    r"""
    Right-hand side of
        d_t (b h)   + d_r (b h u)                        = 0,
        d_t (b h u) + d_r (b (h u^2 + g h^2/2))          = g h^2 b' / 2,
    with b(r) = r, b' = 1: the varying-width form of the radial SWE ([MK14]
    eqs. (1)-(2) with z = 0; [ZR00] eqs. (19)-(20)).  Faces at r = 0 (zero
    measure, no flux) and r = R_OUT (solid wall: zero discharge, pressure g h^2/2).
    """
    n = h.size
    Fh = np.zeros(n + 1)
    Fm = np.zeros(n + 1)
    Fh[1:n], Fm[1:n] = hll_swe(h[:-1], m[:-1], h[1:], m[1:], g)
    Fh[n] = 0.0
    Fm[n] = 0.5 * g * h[-1] ** 2
    dh = -(rf[1:] * Fh[1:] - rf[:-1] * Fh[:-1]) / (rc * dr)
    src = 0.5 * g * h ** 2 * dr
    dm = (-(rf[1:] * Fm[1:] - rf[:-1] * Fm[:-1]) + src) / (rc * dr)
    return dh, dm


def mass_e2(h, rc, dr):
    return float(np.sum(h * rc) * dr)


def front_e2(h, rc):
    """Front radius = argmax |d h/d r| (the detector used by the network run)."""
    return float(rc[int(np.argmax(np.abs(np.gradient(h, rc))))])


# ============================================================================
# 4. E3: 1D intrinsic scheme of Bachini & Putti on the parabolic bed
# ============================================================================
def grid_e3(n):
    xf = np.linspace(0.0, L_E3, n + 1)
    xc = 0.5 * (xf[:-1] + xf[1:])
    return xc, xf


def u0_e3(xc):
    return np.where(xc < DAM_X, ETA_UP, ETA_DN), np.zeros_like(xc)


def e3_faces(eta, q, xc, xf, zface="max", g=G):
    r"""
    Fluxes and folded sources on all faces of the 1D intrinsic scheme.

    Returns (Fh, Fm, srcL, srcR, gam_f, A_f), each of length n+1, where
      * Fh, Fm are the HLL fluxes in the +x sense;
      * srcL[f], srcR[f] are the folded geometric sources  -g (eta*)^2 gam_f^2/2
        seen by the cell on the left/right of face f (identical expressions; the
        flux and the source then cancel identically at rest, which is discrete
        well-balance, [BP20] (3.14)-(3.16));
      * gam_f, A_f are the edge metric coefficients.

    Reconstructions (verbatim [BP20] section 3.2/3.3 in their 1D reduction):
      z_f  = max{z_L, z_R, z(x_f)} if zface='max'  ([A04], needed for exactness
             at dry shores)  else  z(x_f);
      eta* = max(0, (gamma_k eta_k + z_k - z_f)/gam_f)          [BP20] (3.16)
      q*   = q_k [1 - (x_f - x_k) Gamma^1_11(x_k)]              [BP20] (3.12)
    """
    n = eta.size
    gf = gam3(xf)
    Af = 1.0 / gf
    zf = z3(xf).copy()
    if zface == "max" and n > 1:
        zi = z3(xc)
        zf[1:n] = np.maximum(zf[1:n], np.maximum(zi[:-1], zi[1:]))

    # per-face left/right states, index f = face index (0 .. n)
    eL = np.zeros(n + 1)
    qL = np.zeros(n + 1)
    eR = np.zeros(n + 1)
    qR = np.zeros(n + 1)

    if n > 1:
        eL[1:n] = np.maximum(0.0, (gam3(xc[:-1]) * eta[:-1] + z3(xc[:-1]) - zf[1:n]) / gf[1:n])
        eR[1:n] = np.maximum(0.0, (gam3(xc[1:]) * eta[1:] + z3(xc[1:]) - zf[1:n]) / gf[1:n])
        qL[1:n] = q[:-1] * (1.0 - (xf[1:n] - xc[:-1]) * christoffel3(xc[:-1]))
        qR[1:n] = q[1:] * (1.0 - (xf[1:n] - xc[1:]) * christoffel3(xc[1:]))

    # x = 0: solid wall.  Mirror ghost cell to the left of face 0: same metric and
    # bed (so the same reconstructed depth), opposite discharge.
    eL[0] = np.maximum(0.0, (gam3(xc[0]) * eta[0] + z3(xc[0]) - zf[0]) / gf[0])
    eR[0] = eL[0]
    qR[0] = q[0] * (1.0 - (xf[0] - xc[0]) * christoffel3(xc[0]))
    qL[0] = -qR[0]
    # x = L: free outflow.  Transmissive ghost: copy the interior reconstruction.
    eL[n] = np.maximum(0.0, (gam3(xc[-1]) * eta[-1] + z3(xc[-1]) - zf[n]) / gf[n])
    eR[n] = eL[n]
    qL[n] = q[-1] * (1.0 - (xf[n] - xc[-1]) * christoffel3(xc[-1]))
    qR[n] = qL[n]

    # Dry-state closure (see H_DRY): an edge whose reconstructed depth vanishes
    # carries no discharge, which is what keeps the edge velocity v = q/(gamma
    # eta) bounded through v = 0 and makes the HLL wave speeds finite there.
    qL = np.where(eL > H_DRY, qL, 0.0)
    qR = np.where(eR > H_DRY, qR, 0.0)

    Fh, Fm = hll_intrinsic(eL, qL, eR, qR, g, gf, Af)
    srcL = -0.5 * g * eL ** 2 * gf ** 2
    srcR = -0.5 * g * eR ** 2 * gf ** 2
    return Fh, Fm, srcL, srcR, gf, Af


def rhs_e3(eta, q, xc, xf, dx, zface="max", g=G, sink=None, christoffel=True):
    r"""
    Right-hand side of the 1D intrinsic system, in the reduced ("pre-balanced")
    form actually discretised here:

        d_t eta + (1/A) d_x (A q)                                     = 0
        d_t q   + (1/A) d_x (A K) + Gamma K + g eta gamma^2 d_x Z     = 0

    with  K = q^2/eta,  Z = gamma*eta + z  (the physical free-surface height),
    A = h_(1) = 1/gamma,  Gamma = Gamma^1_11 = A'/A,  surface measure ds = A dx.
    This is exactly the x2-independent reduction of the ISWE, [BP20] eq. (2.18)
    with (3.2)-(3.3); the algebra is verified identically in audit 6.4.

    Three pieces form the momentum update
    U^{k+1}_i = U^k_i - (dt/A_i dx) sum_f [ Fm_f + src_f ] - dt (Gamma K)_i :

      * (1/A) d_x(A K)  -->  the HLL flux of A K on the frozen edge metric;
      * g eta gamma^2 d_x Z  -->  NOT a separate term: it is produced by the
        hydrostatic reconstruction (3.16), because the edge state obeys
        eta* gamma_f = gamma_i eta_i + z_i - z_f, so that the pressure part of
        the flux minus the folded source gives
        (g/2)eta_f^2 gamma_f^2 - (g/2)(eta* gamma_f)^2 = g eta_f gamma_f Z'_f (x_f-x_i)
        and the flux difference reproduces g eta gamma^2 d_x Z to O(dx) while
        cancelling identically at rest ([BP20] (3.14)-(3.18));
      * Gamma K is a genuine NON-CONSERVATIVE product: it comes from the
        Christoffel contribution Gamma^j_{ik} F^{ik} of the tensor divergence
        (2.6), and unlike the pressure it is folded nowhere.  It is added as an
        explicit cell source.  Dropping it (as a "flux only" reading of (3.5)
        would) leaves an O(1) residual in the momentum equation that does NOT
        vanish under refinement; reinstating it restores clean first-order
        consistency (audit 6.4) while keeping the lake at rest exact, since
        K = 0 there.

    If `sink` is a list, the outlet discharge used by this evaluation is
    appended to it, so that the mass budget can be closed with exactly the
    fluxes that entered the update.

    `christoffel=False` drops the Gamma K cell source.  It exists ONLY so that
    audit 6.4 can demonstrate the term is necessary (the residual then stalls at
    O(1) instead of converging); it is never used by the solver.
    """
    Fh, Fm, srcL, srcR, _, _ = e3_faces(eta, q, xc, xf, zface, g)
    if sink is not None:
        sink.append(float(Fh[-1]))
    area = A3(xc) * dx
    deta = -(Fh[1:] - Fh[:-1]) / area
    dq = -((Fm[1:] + srcL[1:]) - (Fm[:-1] + srcR[:-1])) / area
    # Christoffel source:  -Gamma^1_11 * q^2/eta,  Gamma^1_11 = A'/A
    if christoffel:
        dq -= christoffel3(xc) * q * q / np.maximum(eta, EPS)
    return deta, dq


def mass_e3(eta, xc, dx):
    """Intrinsic mass  int eta ds  with ds = A dx  ([BP20] (3.4))."""
    return float(np.sum(eta * A3(xc)) * dx)


def front_e3(eta, xc):
    return float(xc[int(np.argmax(np.abs(np.gradient(eta, xc))))])


# ============================================================================
# 5. Time integration and drivers
# ============================================================================
def ssp_rk2(state, dt, rhs):
    """Heun (SSP-RK2).  The flux updates telescope, so mass is conserved to roundoff."""
    r0 = rhs(*state)
    y1 = tuple(s + dt * r for s, r in zip(state, r0))
    r1 = rhs(*y1)
    return tuple(s + 0.5 * dt * (a + b) for s, a, b in zip(state, r0, r1))


def cfl_dt_e2(h, m, rc, rf, dr, cfl, g=G):
    c = np.sqrt(g * np.maximum(h, EPS))
    u = np.abs(_u_of(h, m))
    return cfl * dr / max(float(np.max(u + c)), EPS)


def cfl_dt_e3(eta, q, xc, cfl, g=G):
    v = np.abs(np.where(eta > EPS, A3(xc) * q / np.maximum(eta, EPS), 0.0))
    c = np.sqrt(g * np.maximum(eta, EPS) * gam3(xc))
    return cfl * (xc[1] - xc[0]) / max(float(np.max(v + c)), EPS)


def run_e2(n=4000, tend=T_E2, cfl=0.4, snap_every=0.1, verbose=False, g=G):
    """Advance E2; snapshots land exactly on multiples of snap_every."""
    rc, rf = grid_e2(n)
    dr = rf[1] - rf[0]
    h, m = u0_e2(rc)
    m0 = mass_e2(h, rc, dr)
    snaps = {0.0: (h.copy(), m.copy())}
    t, nstep, next_snap = 0.0, 0, snap_every
    t0 = time.time()
    rhs = lambda hh, mm: rhs_e2(hh, mm, rc, rf, dr, g)
    while t < tend - 1e-12:
        target = min(next_snap, tend)
        dt = min(cfl_dt_e2(h, m, rc, rf, dr, cfl, g), target - t)
        h, m = ssp_rk2((h, m), dt, rhs)
        t += dt
        nstep += 1
        if abs(t - target) < 1e-12:
            snaps[round(t, 6)] = (h.copy(), m.copy())
            if abs(target - next_snap) < 1e-12:
                next_snap += snap_every
    ts = sorted(snaps)
    hh = np.array([snaps[k][0] for k in ts])
    mm = np.array([snaps[k][1] for k in ts])
    mass = np.array([mass_e2(hh[k], rc, dr) for k in range(len(ts))])
    front = np.array([front_e2(hh[k], rc) for k in range(len(ts))])
    if verbose:
        print("   E2 n=%d steps=%d dt_last=%.3e wall=%.1fs max mass drift=%.3e"
              % (n, nstep, dt, time.time() - t0,
                 float(np.max(np.abs(mass - m0)) / m0)))
    return dict(case="e2", n=float(n), r=rc, rf=rf, t=np.array(ts), h=hh, m=mm,
                u=mm / np.maximum(hh, EPS), mass=mass, front=front,
                mass0=m0, steps=float(nstep), wall=time.time() - t0,
                cfl=cfl, tend=tend, g=g)


def run_e3(n=4000, tend=T_E3, cfl=0.4, snap_every=0.1, zface="max",
           verbose=False, g=G):
    """Advance E3; records the mass and the outlet discharge at every step."""
    xc, xf = grid_e3(n)
    dx = xf[1] - xf[0]
    eta, q = u0_e3(xc)
    m0 = mass_e3(eta, xc, dx)
    snaps = {0.0: (eta.copy(), q.copy())}
    t, nstep, next_snap = 0.0, 0, snap_every
    log = []                      # outlet flux of every rhs call (two per step)
    cum = 0.0                     # integral of the outlet discharge, RK2 weights
    cum_t, cum_q, m_all = [0.0], [0.0], [m0]
    snap_q = [0.0]
    t0 = time.time()
    rhs = lambda a, b: rhs_e3(a, b, xc, xf, dx, zface, g, sink=log)
    while t < tend - 1e-12:
        target = min(next_snap, tend)
        dt = min(cfl_dt_e3(eta, q, xc, cfl, g), target - t)
        k0 = len(log)
        eta, q = ssp_rk2((eta, q), dt, rhs)
        q = np.where(eta > H_DRY, q, 0.0)           # dry cells carry no momentum
        cum += dt * 0.5 * (log[k0] + log[k0 + 1])   # exactly the RK2 flux average
        t += dt
        nstep += 1
        cum_t.append(t)
        cum_q.append(cum)
        m_all.append(mass_e3(eta, xc, dx))
        if abs(t - target) < 1e-12:
            snaps[round(t, 6)] = (eta.copy(), q.copy())
            Fh, _, _, _, _, _ = e3_faces(eta, q, xc, xf, zface, g)
            snap_q.append(float(Fh[-1]))
            if abs(target - next_snap) < 1e-12:
                next_snap += snap_every
    ts = sorted(snaps)
    ee = np.array([snaps[k][0] for k in ts])
    qq = np.array([snaps[k][1] for k in ts])
    mass = np.array([mass_e3(ee[k], xc, dx) for k in range(len(ts))])
    front = np.array([front_e3(ee[k], xc) for k in range(len(ts))])
    if verbose:
        print("   E3 n=%d steps=%d dt_last=%.3e wall=%.1fs mass %.6f -> %.6f"
              % (n, nstep, dt, time.time() - t0, mass[0], mass[-1]))
    return dict(case="e3", n=float(n), x=xc, xf=xf, t=np.array(ts), eta=ee, q=qq,
                zeta=z3(xc)[None, :] + gam3(xc)[None, :] * ee,
                mass=mass, front=front, mass0=m0, steps=float(nstep),
                wall=time.time() - t0, cfl=cfl, tend=tend, g=g,
                t_q=np.array(cum_t), int_q_out=np.array(cum_q),
                m_all=np.array(m_all), q_out=np.array(snap_q))


# ============================================================================
# 6. Audit battery
# ============================================================================
def exact_dam_break(hL, hR, uL=0.0, uR=0.0, g=G, it=200):
    r"""
    Exact solution of the wet-bed 1D SWE Riemann problem ([T01] ch. 4).  The
    star depth solves  uL + PhiL(h) = uR + PhiR(h)  with

        Phi_K(h) = 2 (sqrt(g hK) - sqrt(g h))                     if h <= hK
        Phi_K(h) = -(h - hK) sqrt(g (h + hK)/(2 h hK))            if h >  hK   (K=L)
        Phi_K(h) = +(h - hK) sqrt(g (h + hK)/(2 h hK))            if h >  hK   (K=R)

    (the minus in PhiL is the left wave, the plus in PhiR the right wave; note
    uL + PhiL and uR + PhiR must agree).  The left-hand side is monotonically
    decreasing in h, so a bracket [0, hhi] with hhi doubled is bisected to
    machine precision.  Also returns the wave speeds and the wave types.
    """
    def phil(h):
        if h <= hL:
            return 2.0 * (np.sqrt(g * hL) - np.sqrt(g * h))
        return -(h - hL) * np.sqrt(g * (h + hL) / (2.0 * h * hL))

    def phir(h):
        if h <= hR:
            return 2.0 * (np.sqrt(g * hR) - np.sqrt(g * h))
        return (h - hR) * np.sqrt(g * (h + hR) / (2.0 * h * hR))

    def F(h):
        return (uL + phil(h)) - (uR + phir(h))

    lo = 1e-12
    hi = max(hL, hR)
    while F(hi) > 0.0:
        hi *= 2.0
        if hi > 1e8:
            raise RuntimeError("dam-break bracket failed")
    for _ in range(it):
        mid = 0.5 * (lo + hi)
        if F(mid) > 0.0:
            lo = mid
        else:
            hi = mid
    hstar = 0.5 * (lo + hi)
    ustar = 0.5 * ((uL + phil(hstar)) + (uR + phir(hstar)))
    left_rare = hstar <= hL
    right_rare = hstar <= hR
    sL = uL - np.sqrt(g * hL) if left_rare else (hstar * ustar - hL * uL) / (hstar - hL)
    sLt = ustar - np.sqrt(g * hstar) if left_rare else sL
    sRt = ustar + np.sqrt(g * hstar) if right_rare else \
        (hstar * ustar - hR * uR) / (hstar - hR)
    sR = uR + np.sqrt(g * hR) if right_rare else sRt
    return dict(h=hstar, u=ustar, sL=sL, sLt=sLt, sRt=sRt, sR=sR,
                left_rare=left_rare, right_rare=right_rare)


def exact_profile(x, t, hL, hR, uL=0.0, uR=0.0, g=G):
    """Exact depth of the plane dam break, discontinuity at x = 0, t > 0."""
    s = exact_dam_break(hL, hR, uL, uR, g)
    xi = x / max(t, EPS)
    h = np.empty_like(x)
    left = xi <= s["sL"]
    star = (xi > s["sLt"]) & (xi <= s["sRt"])
    right = xi > s["sR"]
    h[left] = hL
    h[star] = s["h"]
    h[right] = hR
    if s["left_rare"]:
        m = (xi > s["sL"]) & (xi <= s["sLt"])
        h[m] = (uL + 2.0 * np.sqrt(g * hL) - xi[m]) ** 2 / (9.0 * g)
    else:
        h[left] = hL
    if s["right_rare"]:
        m = (xi > s["sRt"]) & (xi <= s["sR"])
        h[m] = (2.0 * np.sqrt(g * hR) - uR + xi[m]) ** 2 / (9.0 * g)
    return h, s


def _run_planar(n, tend, cfl=0.4, g=G, xlo=-10.0, xhi=10.0):
    """Plane dam break h=2 (x<0) / 1 (x>0) on a flat bed, walls far outside."""
    xf = np.linspace(xlo, xhi, n + 1)
    xc = 0.5 * (xf[:-1] + xf[1:])
    dx = xf[1] - xf[0]
    h = np.where(xc < 0.0, 2.0, 1.0)
    m = np.zeros_like(xc)

    def rhs(hh, mm):
        n_ = hh.size
        Fh = np.zeros(n_ + 1)
        Fm = np.zeros(n_ + 1)
        Fh[1:n_], Fm[1:n_] = hll_swe(hh[:-1], mm[:-1], hh[1:], mm[1:], g)
        Fh[0], Fm[0] = Fh[1], Fm[1]
        Fh[n_], Fm[n_] = Fh[n_ - 1], Fm[n_ - 1]
        return -(Fh[1:] - Fh[:-1]) / dx, -(Fm[1:] - Fm[:-1]) / dx

    t = 0.0
    while t < tend - 1e-12:
        c = np.sqrt(g * np.maximum(h, EPS))
        dt = min(cfl * dx / max(float(np.max(np.abs(_u_of(h, m)) + c)), EPS),
                 tend - t)
        h, m = ssp_rk2((h, m), dt, rhs)
        t += dt
    return xc, h


def audit_planar(nlist=(500, 1000, 2000, 4000), tend=0.5):
    """6.1  Exact dam-break verification: MAE, relative L1 and its observed order."""
    ref = exact_dam_break(2.0, 1.0)
    hs, us = ref["h"], ref["u"]
    print("  6.1  exact plane dam break hL=2, hR=1, u=0:  h*=%.6f  u*=%.6f" % (hs, us))
    print("       left rarefaction %.6f -> %.6f, right shock S=%.6f m/s"
          % (ref["sL"], ref["sLt"], ref["sRt"]))
    rows, prev = [], None
    print("       %6s %14s %14s %8s" % ("n", "MAE (m)", "rel L1", "order"))
    for n in nlist:
        xc, h = _run_planar(n, tend)
        he, _ = exact_profile(xc, tend, 2.0, 1.0)
        dx = xc[1] - xc[0]
        mae = float(np.sum(np.abs(h - he)) * dx / (xc[-1] - xc[0]))
        rel = float(np.sum(np.abs(h - he)) / np.sum(np.abs(he)))
        order = "" if prev is None else "%8.2f" % (-np.log(rel / prev) / np.log(2.0))
        print("       %6d %14.4e %14.4e %s" % (n, mae, rel, order))
        rows.append((n, mae, rel))
        prev = rel
    return rows, ref


def audit_e2_conservation(n=4000, tend=T_E2):
    """6.2  E2 mass conservation; front speed against the exact planar shock."""
    res = run_e2(n=n, tend=tend, verbose=True)
    drift = float(np.max(np.abs(res["mass"] - res["mass0"])) / res["mass0"])
    t, fr = res["t"], res["front"]
    A = np.polyfit(t[1:], fr[1:], 1)
    ref = exact_dam_break(H_IN, H_OUT)
    print("  6.2  E2 mass M(0)=%.10f, max|M(t)-M(0)|/M(0)=%.3e (machine level)"
          % (res["mass0"], drift))
    print("       front radius %.5f -> %.5f m; fitted mean speed %.4f m/s"
          % (fr[0], fr[-1], A[0]))
    print("       exact *plane* shock speed, same data: %.4f m/s; ratio = %.4f"
          "  (circular front must be slower [MK14])" % (ref["sRt"], A[0] / ref["sRt"]))
    return res, drift, float(A[0]), float(ref["sRt"])


def audit_lake_at_rest(nlist=(500, 1000, 2000), H0s=(4.5, 3.0), nsteps=2000):
    """6.3  E3: the analytic rest state z + gamma*eta = H0 must be preserved exactly."""
    out = []
    for H0 in H0s:
        for n in nlist:
            xc, xf = grid_e3(n)
            dx = xf[1] - xf[0]
            eta = np.maximum((H0 - z3(xc)) / gam3(xc), 0.0)
            eta0, q = eta.copy(), np.zeros_like(xc)
            rhs = lambda a, b: rhs_e3(a, b, xc, xf, dx, "max")
            for _ in range(nsteps):
                eta, q = ssp_rk2((eta, q), cfl_dt_e3(eta, q, xc, 0.4), rhs)
            out.append((H0, n, float(np.max(np.abs(eta - eta0))),
                        float(np.max(np.abs(q)))))
    print("  6.3  E3 lake at rest over the parabolic bed, %d steps:" % nsteps)
    print("       %6s %6s %16s %14s" % ("H0", "n", "max|eta-eta0|", "max|q|"))
    for H0, n, de, qe in out:
        print("       %6.2f %6d %16.3e %14.3e" % (H0, n, de, qe))
    return out


def audit_e3_balance(n=4000, tend=T_E3):
    """6.5  E3 mass budget: the loss must equal the integrated outlet discharge.

    `int_q_out` is the RK2-stage-weighted integral of the *actual* outlet flux
    that entered every update, on the same time levels as `t_q`; it equals
    M(0) - M(T) up to roundoff.  (Using the snapshot list `q_out` here would be
    wrong: it only holds the ~T/0.1 snapshot values, not the per-step fluxes.)
    """
    res = run_e3(n=n, tend=tend, verbose=True)
    tt, qo = res["t_q"], res["int_q_out"]
    lose = res["mass0"] - res["mass"][-1]
    m_end = res["m_all"][-1]
    print("  6.5  E3 mass budget: M(0)=%.6f -> M(T)=%.6f (-%.3f%%), peak |Q_out|=%.4f"
          % (res["mass0"], m_end,
             100.0 * (1.0 - m_end / res["mass0"]),
             float(np.max(np.abs(res["q_out"])))))
    print("       int_0^T Q_out dt = %.9f   vs   M(0)-M(T) = %.9f   rel.diff = %.2e"
          % (qo[-1], lose, abs(qo[-1] - lose) / max(abs(lose), EPS)))
    return res, float(qo[-1]), lose


def _e3_smooth(x):
    """Smooth non-trivial state used by audit 6.4 (q != 0, so K != 0)."""
    eta = (4.5 - z3(x)) / gam3(x) + 0.05 * np.cos(0.7 * x)
    q = 0.30 * np.sin(0.5 * x) + 0.12
    return eta, q


def _e3_ref_rhs(xr):
    """Continuous 1D intrinsic momentum residual, two equivalent forms.

    (E) straight from [BP20] (2.18b):  -(1/A) d_x(A F11) - Gamma F11 - S^1
        with F11 = q^2/eta + (g/2)eta^2 gamma^3 and
        S^1 = (g/2)eta^2 gamma^2 d_x gamma + g eta gamma^2 d_x z;
    (T) reduced form:  -d_x K - 2 Gamma K - g eta gamma^2 d_x Z,  Z = gamma eta + z.
    Their agreement is an algebra self-check.
    """
    eta, q = _e3_smooth(xr)
    d = lambda f: np.gradient(f, xr, edge_order=2)
    A, Ap = A3(xr), A3p(xr)
    gam, gamp, zp = gam3(xr), gam3p(xr), z3p(xr)
    Gam = Ap / A
    K = q * q / np.where(eta > 1e-12, eta, 1e-12)
    F11 = K + 0.5 * G * eta ** 2 * gam ** 3
    rhs_E = -(d(A * F11) / A + Gam * F11
              + 0.5 * G * eta ** 2 * gam ** 2 * gamp + G * eta * gam ** 2 * zp)
    Z = gam * eta + z3(xr)
    rhs_T = -(d(K) + 2.0 * Gam * K + G * eta * gam ** 2 * d(Z))
    return rhs_E, rhs_T


def audit_e3_consistency(nlist=(200, 400, 800, 1600, 3200)):
    """6.4  Consistency of the frozen-metric flux + folded source + Christoffel.

    A smooth state with q != 0 is pushed through the *production* operator
    rhs_e3; in the interior window 2 <= x <= 8 its momentum output must converge
    at first order to the differential residual of the 1D intrinsic system,
    evaluated on a 400001-point grid.  The window excludes the two boundary
    faces, where the wall / outflow closures (not the interior scheme) dominate.

    The third and fourth columns repeat the measurement with the Christoffel
    source switched off: the error then stalls at O(1) and does not converge.
    This is the regression guard for the fix noted in rhs_e3.
    """
    xr = np.linspace(1.0, 9.0, 400001)
    rE, rT = _e3_ref_rhs(xr)
    print("  6.4  consistency of the 1D intrinsic scheme, smooth state with q != 0")
    print("       algebra self-check  max|rhs_E - rhs_T| = %.2e"
          % float(np.max(np.abs(rE - rT))))
    print("       %6s %14s %14s %8s %8s %10s"
          % ("n", "L1(with GamK)", "rel.L1", "order", "", "L1(no GamK)"))
    prev = prevn = None
    for n in nlist:
        xc, xf = grid_e3(n)
        dx = xf[1] - xf[0]
        eta_c, q_c = _e3_smooth(xc)
        sel = (xc >= 2.0) & (xc <= 8.0)
        ref = np.interp(xc, xr, rE)
        scale = float(np.mean(np.abs(ref[sel])))
        _, dq = rhs_e3(eta_c, q_c, xc, xf, dx, "max")
        _, dqn = rhs_e3(eta_c, q_c, xc, xf, dx, "max", christoffel=False)
        e = float(np.mean(np.abs(dq[sel] - ref[sel])))
        en = float(np.mean(np.abs(dqn[sel] - ref[sel])))
        order = "" if prev is None else "%8.2f" % (-np.log(e / prev) / np.log(2.0))
        ordern = "" if prevn is None else "%8.2f" % (-np.log(en / prevn) / np.log(2.0))
        print("       %6d %14.4e %14.4e %s %s %10.4e"
              % (n, e, e / scale, order, ordern, en))
        prev, prevn = e, en
    return None


def audit_selfconvergence(case="e2", nlist=(1000, 2000, 4000, 8000), tend=None):
    """6.6  Self-convergence of the two reference solutions (observed order)."""
    tend = T_E2 if case == "e2" else T_E3
    snap = tend / 10.0
    sols = {}
    for n in nlist:
        sols[n] = (run_e2(n=n, tend=tend, snap_every=snap) if case == "e2"
                   else run_e3(n=n, tend=tend, snap_every=snap))
    print("  6.6  %s grid convergence at t = %.2f s (relative L1 against the next finer grid)"
          % (case.upper(), tend))
    print("       %6s %14s %14s %10s %8s" % ("n", "relL1(h)", "d(front) m", "relL1(f)", "order"))
    prev = None
    for k in range(len(nlist) - 1):
        nc, nf = nlist[k], nlist[k + 1]
        if case == "e2":
            a = sols[nc]["h"][-1]
            b = np.interp(sols[nc]["r"], sols[nf]["r"], sols[nf]["h"][-1])
            fref = sols[nf]["front"][-1]
            fcur = sols[nc]["front"][-1]
            L = R_OUT_E2
        else:
            a = sols[nc]["eta"][-1]
            b = np.interp(sols[nc]["x"], sols[nf]["x"], sols[nf]["eta"][-1])
            fref = sols[nf]["front"][-1]
            fcur = sols[nc]["front"][-1]
            L = L_E3
        e = float(np.sum(np.abs(a - b)) / max(np.sum(np.abs(b)), EPS))
        dfr = abs(fcur - fref) / L
        order = "" if prev is None else "%8.2f" % (-np.log(e / prev) / np.log(2.0))
        print("       %6d %14.4e %14.4e %10.2e %s" % (nc, e, abs(fcur - fref), dfr, order))
        prev = e
    return sols


def audit_e2_vs_2d(n=300, tend=0.4, half=7.0, cfl=0.3):
    """6.7  Independent 2D Cartesian cross-check of the radial reduction.

    A separate, deliberately unspecialised solver: the plain 2D SWE on a
    Cartesian grid with the full 2D HLL flux (`hll_swe_2d`, so the cross
    momentum fluxes are present) and solid walls on all four sides.  It knows
    nothing about radial symmetry.  It must reproduce the 1D radial solution of
    `run_e2` in the annulus 1 m < r < 5.5 m, which is the independent check that
    the b(r) = r reduction of section 3 is the right one.
    """
    m = 2 * n
    xf = np.linspace(-half, half, m + 1)
    xc = 0.5 * (xf[:-1] + xf[1:])
    dx = xf[1] - xf[0]
    X, Y = np.meshgrid(xc, xc, indexing="ij")
    h = np.where(X ** 2 + Y ** 2 < DAM_R ** 2, H_IN, H_OUT)
    mx = np.zeros_like(h)
    my = np.zeros_like(h)

    def rhs(hh, mx_, my_):
        # x-faces
        Fhx = np.zeros((m + 1, m)); Fmx = np.zeros((m + 1, m)); Fmyx = np.zeros((m + 1, m))
        (Fhx[1:m], Fmx[1:m], Fmyx[1:m]) = hll_swe_2d(
            np.stack([hh[:-1], hh[1:]], -1), np.stack([mx_[:-1], mx_[1:]], -1),
            np.stack([my_[:-1], my_[1:]], -1), 1.0, 0.0)
        Fhx[0], Fhx[m] = 0.0, 0.0                       # solid walls
        Fmx[0], Fmx[m] = 0.5 * G * hh[0] ** 2, 0.5 * G * hh[-1] ** 2
        Fmyx[0], Fmyx[m] = 0.0, 0.0
        # y-faces
        Fhy = np.zeros((m, m + 1)); Fmxy = np.zeros((m, m + 1)); Fmy = np.zeros((m, m + 1))
        (Fhy[:, 1:m], Fmxy[:, 1:m], Fmy[:, 1:m]) = hll_swe_2d(
            np.stack([hh[:, :-1], hh[:, 1:]], -1), np.stack([mx_[:, :-1], mx_[:, 1:]], -1),
            np.stack([my_[:, :-1], my_[:, 1:]], -1), 0.0, 1.0)
        Fhy[:, 0], Fhy[:, m] = 0.0, 0.0
        Fmxy[:, 0], Fmxy[:, m] = 0.0, 0.0
        Fmy[:, 0], Fmy[:, m] = 0.5 * G * hh[:, 0] ** 2, 0.5 * G * hh[:, -1] ** 2
        return (-((Fhx[1:] - Fhx[:-1]) + (Fhy[:, 1:] - Fhy[:, :-1])) / dx,
                -((Fmx[1:] - Fmx[:-1]) + (Fmxy[:, 1:] - Fmxy[:, :-1])) / dx,
                -((Fmyx[1:] - Fmyx[:-1]) + (Fmy[:, 1:] - Fmy[:, :-1])) / dx)

    t, step = 0.0, 0
    while t < tend - 1e-12:
        c = np.sqrt(G * np.maximum(h, EPS))
        sp = np.sqrt((mx / np.maximum(h, EPS)) ** 2 + (my / np.maximum(h, EPS)) ** 2)
        dt = min(cfl * dx / max(float(np.max(sp + c)), EPS) / 2.0, tend - t)
        h, mx, my = ssp_rk2((h, mx, my), dt, rhs)
        t += dt
        step += 1
    ref = run_e2(n=8 * n, tend=tend, snap_every=tend)
    r = np.sqrt(X ** 2 + Y ** 2)
    band = (r > 1.0) & (r < 5.5)
    h1d = np.interp(r[band], ref["r"], ref["h"][-1])
    rel = float(np.sum(np.abs(h[band] - h1d)) / np.sum(np.abs(h1d)))
    print("  6.7  Cartesian 2D FV %dx%d (dx=%.4f, %d steps) vs radial solver, t=%.2f s"
          % (m, m, dx, step, tend))
    print("       2D h range [%.4f, %.4f] m  (physical band is [1, 2])"
          % (float(h.min()), float(h.max())))
    print("       relative L1 for 1 m < r < 5.5 m: %.4e, max|dh| = %.4e m"
          % (rel, float(np.max(np.abs(h[band] - h1d)))))
    return rel


# ============================================================================
# 7. Command line
# ============================================================================
def _save(case, res, outdir):
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "%s_n%d.npz" % (case, int(res["n"])))
    d = {k: v for k, v in res.items()
         if isinstance(v, (np.ndarray, float, int))}
    np.savez_compressed(path, **d)
    print("wrote %s (%.1f MB)" % (path, os.path.getsize(path) / 1e6))
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description="1D reference solutions for E2/E3")
    ap.add_argument("--case", choices=["e2", "e3"], default=None)
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--cfl", type=float, default=0.4)
    ap.add_argument("--tend", type=float, default=None)
    ap.add_argument("--snap", type=float, default=0.1)
    ap.add_argument("--zface", choices=["max", "exact"], default="max")
    ap.add_argument("--out", default="out")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--with-2d", action="store_true")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args(argv)

    if a.verify:
        q = a.quick
        print("=" * 78)
        print("swe_ref.py audit battery   numpy %s" % np.__version__)
        print("=" * 78)
        audit_planar(nlist=(500, 1000) if q else (500, 1000, 2000, 4000))
        print()
        audit_e2_conservation(n=1000 if q else 4000)
        print()
        audit_lake_at_rest(nlist=(500,) if q else (500, 1000, 2000), nsteps=500 if q else 2000)
        print()
        audit_e3_consistency(nlist=(200, 400) if q else (200, 400, 800, 1600, 3200))
        print()
        audit_e3_balance(n=1000 if q else 4000)
        print()
        audit_selfconvergence("e2", (1000, 2000, 4000) if q else (1000, 2000, 4000, 8000))
        print()
        audit_selfconvergence("e3", (1000, 2000, 4000) if q else (1000, 2000, 4000, 8000))
        if a.with_2d:
            print()
            audit_e2_vs_2d(n=150 if q else 300, tend=0.4)
        print("=" * 78)
        return 0

    if a.case == "e2":
        res = run_e2(n=a.n, tend=a.tend or T_E2, cfl=a.cfl, snap_every=a.snap,
                     verbose=True)
        _save("e2", res, a.out)
    elif a.case == "e3":
        res = run_e3(n=a.n, tend=a.tend or T_E3, cfl=a.cfl, snap_every=a.snap,
                     zface=a.zface, verbose=True)
        _save("e3", res, a.out)
    else:
        ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
