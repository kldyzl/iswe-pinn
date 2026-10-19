#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
swe_pinn_1d_v2.py —— E3 抛物床溃坝 1D 内蕴 PINN v2（三臂受控实验）

基线：swe_pinn_1d.py（已审计：两装配恒等 1e-14、静湖 2.2e-16、几何闭式 2.4e-12）。
v2 只增不改（除列明的 3 处 bugfix），新增三模式：

  MODE=adam     R1 基线：v1 全部损失 + 去掉单次大 L-BFGS（Adam 3e4 + StepLR）
                ——治层1（L-BFGS 144 步早退），建立归因基线。不含因果加权。
  MODE=viscous  R2a：flux 装配 + 守恒变量人工黏性 ε∂₂²(Aη)、ε∂₂²(Aηu¹)
                + 因果加权（S1, CMAME 421 (2024) 116813）
                ——点态路线的"完成形态"（survey §11.4）。
  MODE=slabfv   R2b：时空 slab 有限体积结构损失（survey §11.3 修正版）
                + 因果加权；移除旧全局守恒罚（slab 求和严格包含它，防双计）。
                守恒成为恒等式：Σ slab 残差 telescopes ⇒ 全域质量收支结构化。

v1 → v2 的 3 处 bugfix：
  B1 evaluate_snapshots：默认设备为 cuda 时 x.numpy() 崩溃（首跑 TypeError 根因）
     ⇒ 统一 .detach().cpu().numpy()。
  B2 训练后立即 torch.save(state_dict)（首跑崩在评估 ⇒ 权重全丢，被迫重训）。
  B3 L-BFGS 改为 opt-in（--lbfgs），默认纯 Adam。

slab-FV 数学（脚本内自审计）：
  守恒形式（对 audited flux 装配乘 A，A=A(x¹) 与 t 无关）：
    质量：  ∂_t m + ∂_x Q = 0,                m = Aη,      Q = Aηu¹
    动量：  ∂_t M + ∂_x(P+Φ) + S = 0,
            M = Aηu¹,  P = Aηu¹²,  Φ = (g/2)γ²η²,  S = A'ηu¹² + g γ η z'
    Φ-分裂恒等式（脚本内机检）：g γ η Z_x ≡ ∂_xΦ + g γ η z'
  slab 残差（cell K_j, 时间 slab [t_{i-1},t_i]；无 ∂ₜ/∂ₓ 求导、无 η 除法
  ⇒ 激波尖峰与干床奇性问题不存在）：
    mass_j,i = m̄_j(t_i) − m̄_j(t_{i-1}) + (1/|K|)∫[Q_R − Q_L]dt
    mom_j,i  = M̄_j(t_i) − M̄_j(t_{i-1}) + (1/|K|)∫[F_R − F_L]dt + (1/|K|)∫∫S dx dt
    （首片 t_0 端用解析初值 ⇒ IC 结构化接入）
  权重约定：wq1[NQ_X] 为 [0,1] 上 GL 权重（Σ=1）⇒ cell 平均 = Σ wq1·f，
  cell 积分 = dx·Σ wq1·f；wqt = (ω/2)·dt 物理时间权重（每 slab Σ=dt）。
  端到端审计：制造光滑解下 slab 装配 ≡ Σ_r wτ_r Σ_q wq1_q·(A·R_flux)
  （autograd 参考积分），逐格逐片对照 <1e-8。
"""
import argparse
import math
import os
import sys
import time

import numpy as np
import torch

# ============================================================================
# 1. 全局配置
# ============================================================================
X0, X1 = 0.0, 10.0
T_END = 1.5
DAM_X = 2.0
H_UP, H_DN = 0.5, 0.2
G_GRAV = 9.81
BED_A, BED_B = 2.0 / 25.0, 2.0 / 25.0

WIDTH, LAYERS = 128, 6
N_SCALE = 10.0

N_INT = 65536
N_IC = 4096
IC_BAND_FRAC = 0.30
N_BC = 2048
N_DENSE_AUDIT = 262144
SEED = 42

W_F, W_I, W_IC_BAND = 1.0, 100.0, 100.0
W_B, W_CONS, W_POS, W_A = 1.0, 0.1, 10.0, 0.0
W_PDE_M = 10.0

H_DW = 1.0e-3
CONS_NT = 16
CONS_DT = 2.0e-3
CONS_SCALE = 0.6
RESIDUAL_ASSEMBLY = "hybrid"              # adam/slabfv 用 hybrid；viscous 强制 flux
LBFGS_MAXITER = 20000
LBFGS_MAXFUN = 80000
N_ITER = 30000                            # v2: Adam 3e4
RECORD_EVERY = 50
LOG_EVERY = 200
N_SNAP_GRID = 401
DEVICE = torch.device("cpu")

# ---- v2 新增 ----
MODE = "adam"                             # adam | viscous | slabfv
EPS_VISC = 0.01
N_SLAB = 30
N_CELL = 128
NQ_X = 32
NQ_T = 6
W_SLAB_M, W_SLAB_P = 10.0, 1.0            # 质量优先（定理 1）
S_M = H_UP                                # 质量残差缩放（η 量纲）
S_P = G_GRAV * H_UP ** 2                  # 动量残差缩放
CAUSAL = True
EPS_CAUSAL0 = 1.0e-2                      # 每 1/3 步 ×10
SLAB_CHUNK = 6                            # pass2 每块的 slab 数（控显存）
CAUSAL_NTIME = 30                         # viscous 点态时间切片数
USE_LBFGS = False


# ============================================================================
# 2. 闭式几何（与 v1 逐字相同）
# ============================================================================
def bed_z(x):
    return (x - X1) ** 2 / 25.0


def bed_z1(x):
    return BED_A * (x - X1)


def bed_z2(x):
    return BED_B


def gam(x):
    return 1.0 / torch.sqrt(1.0 + bed_z1(x) ** 2)


def gam_x(x):
    g = gam(x)
    return -(g ** 3) * bed_z1(x) * bed_z2(x)


def cap_A(x):
    return torch.sqrt(1.0 + bed_z1(x) ** 2)


def cap_A1(x):
    return -gam_x(x) / gam(x) ** 2


def cap_Gamma(x):
    return cap_A1(x) / cap_A(x)


def h_initial(x):
    return torch.where(x < DAM_X,
                       torch.full_like(x, H_UP), torch.full_like(x, H_DN))


# ============================================================================
# 3. 网络
# ============================================================================
class NLAFNet(torch.nn.Module):
    def __init__(self, layers=LAYERS, width=WIDTH, n=N_SCALE):
        super().__init__()
        self.n = float(n)
        self.hids = torch.nn.ModuleList()
        self.slopes = torch.nn.ParameterList()
        d_in = 2
        for _ in range(layers):
            lin = torch.nn.Linear(d_in, width)
            torch.nn.init.kaiming_normal_(lin.weight, nonlinearity="tanh")
            torch.nn.init.zeros_(lin.bias)
            self.hids.append(lin)
            self.slopes.append(torch.nn.Parameter(torch.full((width,), 1.0 / self.n)))
            d_in = width
        self.out = torch.nn.Linear(width, 2)
        torch.nn.init.xavier_uniform_(self.out.weight)
        torch.nn.init.zeros_(self.out.bias)

    def forward(self, X):
        z = torch.stack([
            (X[:, 0] - (0.5 * (X0 + X1))) / (0.5 * (X1 - X0)),
            X[:, 1] / T_END * 2.0 - 1.0,
        ], dim=1)
        for lin, a in zip(self.hids, self.slopes):
            z = torch.tanh(self.n * a * lin(z))
        return self.out(z).unbind(dim=1)

    def slope_recovery(self):
        Dm1 = len(self.slopes)
        denom = sum(torch.exp(a.mean()) / Dm1 for a in self.slopes)
        return 1.0 / denom


class ManufacturedNet(torch.nn.Module):
    """光滑制造解（只用于 slab 端到端审计）。"""

    @staticmethod
    def forward(X):
        x = X[:, 0]
        t = X[:, 1]
        eta = (0.35 + 0.10 * torch.sin(2 * math.pi * x / 10.0 + 0.3)
               * torch.cos(math.pi * t / T_END))
        u1 = 0.10 * torch.sin(math.pi * x / 10.0) * torch.cos(2 * math.pi * t / T_END)
        return eta, u1


# ============================================================================
# 4. 残差
# ============================================================================
def _first_grads(net, X):
    eta, u1 = net(X)
    xg = X
    g_eta = torch.autograd.grad(eta.sum(), xg, create_graph=True)[0]
    g_u1 = torch.autograd.grad(u1.sum(), xg, create_graph=True)[0]
    eta_t, eta_x = g_eta[:, 1], g_eta[:, 0]
    u1_t, u1_x = g_u1[:, 1], g_u1[:, 0]
    return eta, u1, eta_t, eta_x, u1_t, u1_x


def residuals_hybrid(net, X):
    eta, u1, eta_t, eta_x, u1_t, u1_x = _first_grads(net, X)
    x = X[:, 0]
    A = cap_A(x)
    Gam = cap_Gamma(x)
    g = gam(x)
    gx = gam_x(x)
    zx = bed_z1(x)
    Z_x = g * eta_x + eta * gx + zx
    R_mass = eta_t + u1 * eta_x + eta * (u1_x + Gam * u1)
    R_mom = (u1 * R_mass + eta * (u1_t + u1 * u1_x + G_GRAV * g ** 2 * Z_x)
             + Gam * eta * u1 ** 2)
    return R_mass, R_mom, eta, u1


def residuals_flux(net, X):
    eta, u1, eta_t, eta_x, u1_t, u1_x = _first_grads(net, X)
    x = X[:, 0]
    A = cap_A(x)
    Gam = cap_Gamma(x)
    g = gam(x)
    gx = gam_x(x)
    zx = bed_z1(x)
    Z_x = g * eta_x + eta * gx + zx
    q = A * eta * u1
    Kx = A * eta * u1 ** 2
    dq_x = torch.autograd.grad(q.sum(), X, create_graph=True)[0][:, 0]
    dK_x = torch.autograd.grad(Kx.sum(), X, create_graph=True)[0][:, 0]
    R_mass = eta_t + dq_x / A
    qt = eta * u1
    dq_t = torch.autograd.grad(qt.sum(), X, create_graph=True)[0][:, 1]
    R_mom = dq_t + dK_x / A + Gam * eta * u1 ** 2 + G_GRAV * eta * g ** 2 * Z_x
    return R_mass, R_mom, eta, u1


def residuals_visc(net, X, eps=None):
    """flux 装配 + 守恒变量黏性：∂_t m + ∂_x Q = ε∂₂²m；
    ∂_t M + ∂_x(P+Φ) + S = ε∂₂²M（M = Aηu¹ = q ⇒ ∂₂²M = ∂₂²q）。
    边界黏性泄漏 O(ε·∂₁m|_)≈5e-4 ≪ CONS_SCALE，不做渐隐（survey §11.4）。"""
    eps = EPS_VISC if eps is None else eps
    eta, u1, eta_t, eta_x, u1_t, u1_x = _first_grads(net, X)
    x = X[:, 0]
    A = cap_A(x)
    Gam = cap_Gamma(x)
    g = gam(x)
    gx = gam_x(x)
    zx = bed_z1(x)
    Z_x = g * eta_x + eta * gx + zx
    m = A * eta
    q = A * eta * u1
    Kx = A * eta * u1 ** 2
    qt = eta * u1
    dm_x = torch.autograd.grad(m.sum(), X, create_graph=True)[0][:, 0]
    dq_x = torch.autograd.grad(q.sum(), X, create_graph=True)[0][:, 0]
    dK_x = torch.autograd.grad(Kx.sum(), X, create_graph=True)[0][:, 0]
    dq_t = torch.autograd.grad(qt.sum(), X, create_graph=True)[0][:, 1]
    d2m_x = torch.autograd.grad(dm_x.sum(), X, create_graph=True)[0][:, 0]
    d2q_x = torch.autograd.grad(dq_x.sum(), X, create_graph=True)[0][:, 0]
    R_mass = eta_t + dq_x / A - (eps / A) * d2m_x
    R_mom = (dq_t + dK_x / A + Gam * eta * u1 ** 2 + G_GRAV * eta * g ** 2 * Z_x
             - (eps / A) * d2q_x)
    return R_mass, R_mom, eta, u1


def residuals(net, X, assembly=None):
    asm = assembly or RESIDUAL_ASSEMBLY
    if MODE == "viscous":
        return residuals_visc(net, X)
    if asm == "hybrid":
        return residuals_hybrid(net, X)
    if asm == "flux":
        return residuals_flux(net, X)
    raise ValueError("RESIDUAL_ASSEMBLY 必须是 'hybrid' 或 'flux'")


# ============================================================================
# 5. 采样（与 v1 相同）
# ============================================================================
def _lowdisc(n, d, seed):
    try:
        from scipy.stats import qmc
        s = qmc.Sobol(d=d, scramble=True, seed=seed)
        return torch.tensor(s.random(n), dtype=torch.float64)
    except Exception:
        rng = np.random.RandomState(seed)
        return torch.tensor(rng.random_sample((n, d)), dtype=torch.float64)


def _in_box(u):
    return torch.stack([u[:, 0] * (X1 - X0) + X0, u[:, 1] * T_END], dim=1)


def make_points():
    u = _lowdisc(N_INT, 2, SEED + 101)
    Xp = _in_box(u)
    u_ic = _lowdisc(N_IC, 2, SEED + 102)
    n_band = int(N_IC * IC_BAND_FRAC)
    u_band = _lowdisc(n_band, 2, SEED + 103)
    Xi_main = torch.stack([u_ic[:, 0] * (X1 - X0) + X0,
                           torch.zeros(N_IC, dtype=torch.float64)], dim=1)
    Xi_band = torch.stack([DAM_X + (u_band[:, 0] - 0.5) * 0.1,
                           torch.zeros(n_band, dtype=torch.float64)], dim=1)
    Xb0 = torch.stack([torch.full((N_BC,), X0, dtype=torch.float64),
                       _lowdisc(N_BC, 1, SEED + 104)[:, 0] * T_END], dim=1)
    Xb1 = torch.stack([torch.full((N_BC,), X1, dtype=torch.float64),
                       _lowdisc(N_BC, 1, SEED + 105)[:, 0] * T_END], dim=1)
    return Xp, Xi_main, Xi_band, Xb0, Xb1


def dense_audit_points(n):
    u = _lowdisc(n, 2, SEED + 999)
    return _in_box(u)


# ============================================================================
# 6. 守恒罚（v1 逐字保留；slabfv 模式下不调用）
# ============================================================================
_CONS_CACHE = {}


def _cons_times():
    if "_t" not in _CONS_CACHE:
        u = _lowdisc(CONS_NT, 1, SEED + 404)
        t0 = (u[:, 0] * (T_END - 2 * CONS_DT) + CONS_DT).to(DEVICE)
        _CONS_CACHE["_t"] = t0
    return _CONS_CACHE["_t"]


def _cons_grids():
    if "x" not in _CONS_CACHE:
        x = torch.linspace(X0, X1, 401, dtype=torch.float64, device=DEVICE)
        w = torch.full((401,), (X1 - X0) / 400, dtype=torch.float64, device=DEVICE)
        w[0] *= 0.5
        w[-1] *= 0.5
        _CONS_CACHE["x"] = (x, w)
    return _CONS_CACHE["x"]


def mass_conservation_penalty(net):
    (xg, w) = _cons_grids()
    t0s = _cons_times()
    loss = torch.zeros((), dtype=torch.float64, device=DEVICE)
    for t0 in t0s:
        def _M(tc):
            Xc = torch.stack([xg, torch.full_like(xg, tc)], dim=1)
            eta, _ = net(Xc)
            return torch.sum(cap_A(xg) * eta * w)

        def _Q(tc):
            Xc = torch.stack([torch.full((1,), X1, dtype=torch.float64, device=DEVICE),
                              torch.full((1,), tc, dtype=torch.float64, device=DEVICE)],
                             dim=1)
            eta, u1 = net(Xc)
            return (cap_A(Xc[:, 0]) * eta * u1).reshape(())

        dM = (_M(t0 + CONS_DT) - _M(t0 - CONS_DT)) / (2.0 * CONS_DT)
        loss = loss + ((dM + _Q(t0)) / CONS_SCALE) ** 2
    return loss / float(CONS_NT)


# ============================================================================
# 7. slab-FV 机器（v2 新增）
# ============================================================================
_SLAB_CACHE = {}


def _slab_grid():
    """权重约定：wq1[NQ_X] 为 [0,1] 上 GL 权重（Σ=1）⇒ cell 平均 = Σ wq1·f，
    cell 积分 = dx·Σ wq1·f；wqt[NS*NQT] = (ω/2)·dt 物理时间权重（每 slab Σ=dt）。"""
    if "grid" in _SLAB_CACHE:
        return _SLAB_CACHE["grid"]
    from numpy.polynomial.legendre import leggauss
    xe = torch.linspace(X0, X1, N_CELL + 1, dtype=torch.float64, device=DEVICE)
    dx = (X1 - X0) / N_CELL
    xi, wq = leggauss(NQ_X)
    xq_locc = torch.tensor((xi + 1) / 2, dtype=torch.float64, device=DEVICE)
    wq1 = torch.tensor(np.asarray(wq) / 2, dtype=torch.float64, device=DEVICE)
    xg = (xe[:-1].unsqueeze(1) + xq_locc.unsqueeze(0) * dx).reshape(-1)   # [NC*NQ]
    tb = torch.linspace(0.0, T_END, N_SLAB + 1, dtype=torch.float64, device=DEVICE)
    dt = T_END / N_SLAB
    xit, wtt = leggauss(NQ_T)
    tq_locc = torch.tensor((xit + 1) / 2, dtype=torch.float64, device=DEVICE)
    wqt_locc = torch.tensor(np.asarray(wtt) / 2, dtype=torch.float64,
                            device=DEVICE) * dt
    tau = (tb[:-1].unsqueeze(1) + tq_locc.unsqueeze(0) * dt).reshape(-1)  # [NS*NQT]
    wqt = wqt_locc.unsqueeze(0).expand(N_SLAB, NQ_T).reshape(-1)
    _SLAB_CACHE["grid"] = (xe, dx, xg, tb, dt, tau)
    _SLAB_CACHE["wq1"] = wq1
    _SLAB_CACHE["wqt"] = wqt
    _SLAB_CACHE["xg2"] = xg.reshape(N_CELL, NQ_X)
    return _SLAB_CACHE["grid"]


def _ic_cell_means():
    """解析初值 cell 平均 m̄(t_0)、M̄(t_0)（与网络侧同一套 GL 节点 ⇒ 静湖逐位一致）。"""
    if "ic" in _SLAB_CACHE:
        return _SLAB_CACHE["ic"]
    _slab_grid()
    xg2 = _SLAB_CACHE["xg2"]
    wq1 = _SLAB_CACHE["wq1"]
    m0 = torch.sum(wq1.unsqueeze(0) * cap_A(xg2) * h_initial(xg2), dim=1)  # [NC]
    M0 = torch.zeros_like(m0)                                              # u¹(0)≡0
    _SLAB_CACHE["ic"] = (m0, M0)
    return (m0, M0)


def slab_cell_means(net, levels=None):
    """时间边界层上的 m̄、M̄（cell 平均）。levels 缺省=全部 NS+1 层。
    返回 (mbar, Mbar)：[n_level, NC]。保留计算图（可反传）。"""
    (xe, dx, xg, tb, dt, tau) = _slab_grid()
    wq1 = _SLAB_CACHE["wq1"]
    lv = list(range(N_SLAB + 1)) if levels is None else list(levels)
    mbar, Mbar = [], []
    CH = 4
    for s in range(0, len(lv), CH):
        grp = lv[s:s + CH]
        Xc = torch.stack([
            xg.unsqueeze(0).expand(len(grp), -1).reshape(-1),
            tb[grp].unsqueeze(1).expand(-1, xg.shape[0]).reshape(-1),
        ], dim=1)
        eta, u1 = net(Xc)
        A = cap_A(Xc[:, 0])
        m = (A * eta).reshape(len(grp), N_CELL, NQ_X)
        M = (A * eta * u1).reshape(len(grp), N_CELL, NQ_X)
        mbar.append(torch.sum(wq1.reshape(1, 1, NQ_X) * m, dim=2))
        Mbar.append(torch.sum(wq1.reshape(1, 1, NQ_X) * M, dim=2))
    return torch.cat(mbar, dim=0), torch.cat(Mbar, dim=0)


def slab_edge_fluxes(net, slabs=None):
    """各 slab 时间 GL 节点处 cell 边上的 Q 与 P+Φ。返回 [n_slab, NQ_T, NC+1]。"""
    (xe, dx, xg, tb, dt, tau) = _slab_grid()
    sl = list(range(N_SLAB)) if slabs is None else list(slabs)
    CH = 10
    Q_all, F_all = [], []
    for s in range(0, len(sl), CH):
        grp = sl[s:s + CH]
        n_sl = len(grp)
        xe_row = xe.unsqueeze(0).expand(n_sl * NQ_T, -1)
        idx = torch.tensor(grp, device=DEVICE).unsqueeze(1) * NQ_T \
            + torch.arange(NQ_T, device=DEVICE).unsqueeze(0)          # [n_sl, NQT]
        t_row = tau[idx].reshape(-1).unsqueeze(1).expand(-1, N_CELL + 1)
        Xe = torch.stack([xe_row.reshape(-1), t_row.reshape(-1)], dim=1)
        eta, u1 = net(Xe)
        A = cap_A(Xe[:, 0])
        g = gam(Xe[:, 0])
        Q = A * eta * u1
        Fp = A * eta * u1 ** 2 + 0.5 * G_GRAV * g ** 2 * eta ** 2
        Q_all.append(Q.reshape(n_sl, NQ_T, N_CELL + 1))
        F_all.append(Fp.reshape(n_sl, NQ_T, N_CELL + 1))
    return torch.cat(Q_all), torch.cat(F_all)


def slab_source_ints(net, slabs):
    """源项 S = A'ηu¹² + g γ η z' 的 cell 积分 × 时间权重（调用方控制分块）。
    返回 [n_slab, NQ_T, NC]，元素 = wτ_r · dx·Σ_q wq1·S（真积分）。"""
    (xe, dx, xg, tb, dt, tau) = _slab_grid()
    wq1 = _SLAB_CACHE["wq1"]
    wqt2 = _SLAB_CACHE["wqt"].reshape(N_SLAB, NQ_T)
    sl = list(slabs)
    n_sl = len(sl)
    idx = torch.tensor(sl, device=DEVICE).unsqueeze(1) * NQ_T \
        + torch.arange(NQ_T, device=DEVICE).unsqueeze(0)              # [n_sl, NQT]
    t_row = tau[idx]
    Xs = torch.stack([
        xg.unsqueeze(0).expand(n_sl * NQ_T, -1).reshape(-1),
        t_row.reshape(-1).unsqueeze(1).expand(-1, xg.shape[0]).reshape(-1),
    ], dim=1)
    eta, u1 = net(Xs)
    A1 = cap_A1(Xs[:, 0])
    g = gam(Xs[:, 0])
    zx = bed_z1(Xs[:, 0])
    Ssrc = (A1 * eta * u1 ** 2 + G_GRAV * g * eta * zx).reshape(n_sl, NQ_T,
                                                                N_CELL, NQ_X)
    ints = dx * torch.sum(wq1.reshape(1, 1, 1, NQ_X) * Ssrc, dim=3)   # [n_sl,NQT,NC]
    return ints * wqt2[torch.tensor(sl, device=DEVICE)].unsqueeze(2)


def _assemble_slab(mbar, Mbar, Q_e, F_e, src, i, wqt_i, dx):
    """第 i 个 slab 的残差（物理量纲）。mbar/Mbar: [.., NC]；Q_e/F_e: [NQ_T, NC+1]；
    src: [NQ_T, NC]（已含 wτ·dx）。"""
    flux_m = Q_e[:, 1:] - Q_e[:, :-1]
    flux_p = F_e[:, 1:] - F_e[:, :-1]
    rm = mbar - 0 + (1.0 / dx) * torch.sum(wqt_i.unsqueeze(1) * flux_m, dim=0)
    rp = (Mbar + (1.0 / dx) * torch.sum(wqt_i.unsqueeze(1) * flux_p, dim=0)
          + (1.0 / dx) * torch.sum(src, dim=0))
    return rm, rp


def slab_residuals(net, with_grad=True, analytic_ic=True):
    """全部 slab 残差。返回 (res_m [NS,NC], res_p [NS,NC])。
    analytic_ic=True：首片 t_0 端用解析初值（训练用，IC 结构化接入）；
    analytic_ic=False：第 0 层用网络自身值（审计用，纯测装配本身）。"""
    (xe, dx, xg, tb, dt, tau) = _slab_grid()
    ctx = torch.enable_grad() if with_grad else torch.no_grad()
    with ctx:
        mbar, Mbar = slab_cell_means(net)
        Q_e, F_e = slab_edge_fluxes(net)
        if analytic_ic:
            m0, M0 = _ic_cell_means()
            mbar = torch.cat([m0.unsqueeze(0), mbar[1:]], dim=0)
            Mbar = torch.cat([M0.unsqueeze(0), Mbar[1:]], dim=0)
        res_m, res_p = [], []
        for i in range(N_SLAB):
            wqt_i = _SLAB_CACHE["wqt"].reshape(N_SLAB, NQ_T)[i]
            src = slab_source_ints(net, [i]).squeeze(0)
            rm, rp = _assemble_slab(mbar[i + 1], Mbar[i + 1] - 0, Q_e[i], F_e[i],
                                    src, i, wqt_i, dx)
            rm = rm + (- mbar[i])
            rp = rp + (- Mbar[i])
            res_m.append(rm)
            res_p.append(rp)
        return torch.stack(res_m), torch.stack(res_p)


def slab_losses(net, with_grad=True, analytic_ic=True):
    """每 slab 标量损失（归一化后）。返回 list[NS]。"""
    res_m, res_p = slab_residuals(net, with_grad=with_grad, analytic_ic=analytic_ic)
    L = torch.mean((res_m / S_M) ** 2 * W_SLAB_M + (res_p / S_P) ** 2 * W_SLAB_P,
                   dim=1)
    return [L[i] for i in range(N_SLAB)]


def _causal_weights(loss_list, eps_c):
    """w_i = exp(−ε_c Σ_{k<i} L_k)，均值归一 + 下限；全 detach（survey §11.2 风险①）。"""
    with torch.no_grad():
        L = torch.stack([l.detach() if torch.is_tensor(l) else
                         torch.tensor(float(l), dtype=torch.float64,
                                      device=DEVICE) for l in loss_list])
        cum = torch.cumsum(L, dim=0) - L
        w = torch.exp(-eps_c * cum)
        w = w / torch.mean(w)
        w = torch.clamp(w, min=1.0e-2)
    return w


def _eps_causal(it):
    """ε_c 退火：每 1/3 训练步 ×10（起点 EPS_CAUSAL0）。"""
    k = int(3.0 * it / max(N_ITER, 1))
    return EPS_CAUSAL0 * (10.0 ** k)


# ============================================================================
# 8. 损失（模式分发）
# ============================================================================
def total_loss(net, Xp, Xi_main, Xi_band, Xb0, Xb1, it=0):
    """返回 (L_total, L_pde, L_ic, L_bc, L_cons, L_pos, L_out)。
    注意：causal 只用于 viscous（R1 必须是纯基线；slabfv 的 causal 在 train() 的
    slab 两遍法里）。"""
    if MODE == "viscous":
        R_mass, R_mom, _, _ = residuals_visc(net, Xp)
    else:
        R_mass, R_mom, _, _ = residuals(net, Xp)
    resid = W_PDE_M * R_mass ** 2 + R_mom ** 2
    if CAUSAL and MODE == "viscous":
        tsel = torch.clamp((Xp[:, 1] / T_END * CAUSAL_NTIME).long(),
                           max=CAUSAL_NTIME - 1)
        losses = [resid[tsel == i].mean() for i in range(CAUSAL_NTIME)]
        w = _causal_weights(losses, _eps_causal(it))
        L_pde = torch.sum(w * torch.stack(losses)) / CAUSAL_NTIME
    else:
        L_pde = torch.mean(resid)

    eta_i, u1_i = net(Xi_main)
    h0 = h_initial(Xi_main[:, 0])
    L_ic = torch.mean((eta_i - h0) ** 2 + u1_i ** 2)

    eta_b, u1_b = net(Xi_band)
    h0b = h_initial(Xi_band[:, 0])
    L_ic_band = torch.mean((eta_b - h0b) ** 2 + u1_b ** 2)

    eta_w0, u1_w0 = net(Xb0)
    L_wall = torch.mean((cap_A(Xb0[:, 0]) * eta_w0 * u1_w0) ** 2)

    Xg = Xb1.clone().requires_grad_(True)
    eta_o, u1_o = net(Xg)
    f = u1_o - 2.0 * torch.sqrt(torch.clamp(G_GRAV * gam(Xg[:, 0]) * eta_o, min=0.0))
    d1 = torch.autograd.grad(f.sum(), Xg, create_graph=True)[0][:, 0]
    L_out = torch.mean(d1 ** 2)
    L_bc = L_wall + L_out

    eta_p, u1_p = net(Xp)
    w_dw = torch.clamp(torch.relu(H_DW - eta_p) / H_DW, 0.0, 1.0)
    L_pos = torch.mean(torch.relu(-eta_p) ** 2 + w_dw * u1_p ** 2)

    if MODE == "slabfv":
        L_cons = torch.zeros((), dtype=torch.float64, device=DEVICE)  # 已被 slab 取代
    else:
        L_cons = mass_conservation_penalty(net)

    L_total = (W_F * L_pde + W_I * L_ic + W_IC_BAND * L_ic_band + W_B * L_bc
               + W_CONS * L_cons + W_POS * L_pos + W_A * net.slope_recovery())
    return L_total, L_pde, L_ic, L_bc, L_cons, L_pos, L_out


# ============================================================================
# 9. 审计
# ============================================================================
def audit_assembly_identity(net, X):
    Rh_m, Rh_o, _, _ = residuals_hybrid(net, X)
    Rf_m, Rf_o, _, _ = residuals_flux(net, X)
    d1 = float(torch.max(torch.abs(Rh_m - Rf_m)))
    d2 = float(torch.max(torch.abs(Rh_o - Rf_o)))
    print("  [审计:装配] max|hybrid−flux| mass=%.3e mom=%.3e （须 < 1e-9）" % (d1, d2))
    ok = d1 < 1e-9 and d2 < 1e-9
    print("  [审计:装配] %s" % ("通过 ✅" if ok else "失败 ❌"))
    return ok


def audit_geometry():
    x = torch.linspace(0.5, 9.5, 41, dtype=torch.float64, device=DEVICE)
    h = 1e-4
    gp = (gam(x + h) - gam(x - h)) / (2 * h)
    ap = (cap_A(x + h) - cap_A(x - h)) / (2 * h)
    e1 = float(torch.max(torch.abs(gp - gam_x(x))))
    e2 = float(torch.max(torch.abs(ap - cap_A1(x))))
    print("  [审计:几何] max|γ'_闭式−差分|=%.3e  max|A'_闭式−差分|=%.3e （须 <1e-7）"
          % (e1, e2))
    ok = e1 < 1e-7 and e2 < 1e-7
    print("  [审计:几何] %s" % ("通过 ✅" if ok else "失败 ❌"))
    return ok


def audit_ic_mass():
    x = torch.linspace(X0, X1, 40001, dtype=torch.float64, device=DEVICE)
    A = cap_A(x)
    eta0 = h_initial(x)
    w = torch.full_like(x, (X1 - X0) / 40000)
    w[0] *= 0.5
    w[-1] *= 0.5
    M0 = float(torch.sum(A * eta0 * w))
    print("  [审计:IC] M0 闭式 = %.6f（参考质量应一致）" % M0)
    return M0


def audit_static_lake():
    x = torch.linspace(0.2, 9.8, 999, dtype=torch.float64, device=DEVICE)
    H0 = 1.0
    eta_lake = (H0 - bed_z(x)) / gam(x)
    Z = gam(x) * eta_lake + bed_z(x)
    e = float(torch.max(torch.abs(Z - H0)))
    print("  [审计:静湖] max|Z − H₀| = %.3e （闭式恒等）" % e)
    return e < 1e-12


def audit_pressure_split():
    """Φ-分裂恒等式：g γ η Z_x ≡ ∂_x(g/2 γ²η²) + g γ η z'（η=1 常值探针）。"""
    x = torch.linspace(0.2, 9.8, 999, dtype=torch.float64, device=DEVICE)
    xg = x.clone().requires_grad_(True)
    eta1 = torch.ones_like(xg)
    Phi = 0.5 * G_GRAV * gam(xg) ** 2 * eta1 ** 2
    dPhi = torch.autograd.grad(Phi.sum(), xg, create_graph=True)[0]
    lhs = G_GRAV * gam(xg) * eta1 * (gam_x(xg) + bed_z1(xg))   # Z_x = γ_x + z_x
    rhs = dPhi + G_GRAV * gam(xg) * eta1 * bed_z1(xg)
    e = float(torch.max(torch.abs(lhs - rhs)))
    print("  [审计:Φ-分裂] max|gγηZ_x − (∂_xΦ + gγηz')| = %.3e （须 <1e-12）" % e)
    ok = e < 1e-12
    print("  [审计:Φ-分裂] %s" % ("通过 ✅" if ok else "失败 ❌"))
    return ok


def audit_slab_static_lake():
    """静湖（η=(H₀−z)/γ, u¹=0）的 slab 残差应为准机器零（解析抵消 + GL 对三次
    多项式精确）。"""
    H0_L = 1.0

    class LakeNet(torch.nn.Module):
        def forward(self, X):
            return ((H0_L - bed_z(X[:, 0])) / gam(X[:, 0]),
                    torch.zeros_like(X[:, 0]))

    Ls = slab_losses(LakeNet(), with_grad=False, analytic_ic=False)
    mx = max(float(l) for l in Ls)
    print("  [审计:slab静湖] max_i L_slab_i = %.3e （须 <1e-12）" % mx)
    ok = mx < 1e-12
    print("  [审计:slab静湖] %s" % ("通过 ✅" if ok else "失败 ❌"))
    return ok


def audit_slab_manufactured():
    """端到端：制造光滑解下，slab 装配 ≡ Σ_r wτ_r Σ_q wq1_q (A·R_flux)（autograd
    参考积分）。一次校验 m̄/M̄/边通量/Φ-分裂/源项/符号/缩放全部环节。"""
    (xe, dx, xg, tb, dt, tau) = _slab_grid()
    wq1 = _SLAB_CACHE["wq1"]
    wqt2 = _SLAB_CACHE["wqt"].reshape(N_SLAB, NQ_T)
    mnet = ManufacturedNet()
    res_m, res_p = slab_residuals(mnet, with_grad=False, analytic_ic=False)
    Xq = torch.stack([
        xg.unsqueeze(0).expand(N_SLAB * NQ_T, -1).reshape(-1),
        tau.unsqueeze(1).expand(-1, xg.shape[0]).reshape(-1),
    ], dim=1).requires_grad_(True)
    Rm, Ro, _, _ = residuals_flux(mnet, Xq)
    A = cap_A(Xq[:, 0])
    am = (A * Rm).reshape(N_SLAB, NQ_T, N_CELL, NQ_X)
    ao = (A * Ro).reshape(N_SLAB, NQ_T, N_CELL, NQ_X)
    ref_m = torch.sum(wqt2.reshape(N_SLAB, NQ_T, 1, 1) * am
                      * wq1.reshape(1, 1, 1, NQ_X), dim=(1, 3))
    ref_p = torch.sum(wqt2.reshape(N_SLAB, NQ_T, 1, 1) * ao
                      * wq1.reshape(1, 1, 1, NQ_X), dim=(1, 3))
    d1 = float(torch.max(torch.abs(res_m - ref_m)))
    d2 = float(torch.max(torch.abs(res_p - ref_p)))
    scale = max(1.0, float(torch.max(torch.abs(ref_m))),
                float(torch.max(torch.abs(ref_p))))
    print("  [审计:slab制造解] max|装配−autograd参考| mass=%.3e mom=%.3e （须 <1e-8·scale）"
          % (d1, d2))
    ok = d1 < 1e-8 * scale and d2 < 1e-8 * scale
    print("  [审计:slab制造解] %s" % ("通过 ✅" if ok else "失败 ❌"))
    return ok


def sampling_adequacy_report(net, Xp, tag=""):
    Xd = dense_audit_points(min(N_DENSE_AUDIT, 65536)).to(DEVICE).requires_grad_(True)
    Rm_d, Ro_d, _, _ = residuals(net, Xd)
    L_d = float(torch.mean(W_PDE_M * Rm_d ** 2 + Ro_d ** 2))
    Rm_t, Ro_t, _, _ = residuals(net, Xp)
    L_t = float(torch.mean(W_PDE_M * Rm_t ** 2 + Ro_t ** 2))
    ratio = L_d / max(L_t, 1e-300)
    print("  %s L_pde(密点)=%.4e  L_pde(配点)=%.4e  ratio=%.2f （判据 <10）"
          % (tag, L_d, L_t, ratio))
    return ratio


def mass_balance_report(net, tag=""):
    (xg, w) = _cons_grids()
    ts = torch.linspace(CONS_DT, T_END - CONS_DT, 200, dtype=torch.float64,
                        device=DEVICE)
    worst = 0.0
    qmax = 0.0
    for t0 in ts:
        Xc = torch.stack([xg, torch.full_like(xg, t0)], dim=1)
        eta, _ = net(Xc)
        M = float(torch.sum(cap_A(xg) * eta * w))
        Xc2 = torch.stack([torch.full((1,), X1, dtype=torch.float64, device=DEVICE),
                           torch.full((1,), t0, dtype=torch.float64, device=DEVICE)],
                          dim=1)
        eta2, u12 = net(Xc2)
        q = float((cap_A(Xc2[:, 0]) * eta2 * u12).reshape(()))
        Xc3 = torch.stack([xg, torch.full_like(xg, t0 + CONS_DT)], dim=1)
        eta3, _ = net(Xc3)
        M3 = float(torch.sum(cap_A(xg) * eta3 * w))
        Xc4 = torch.stack([xg, torch.full_like(xg, t0 - CONS_DT)], dim=1)
        eta4, _ = net(Xc4)
        M4 = float(torch.sum(cap_A(xg) * eta4 * w))
        dM = (M3 - M4) / (2 * CONS_DT)
        qmax = max(qmax, abs(q))
        worst = max(worst, abs(dM + q))
    ratio = worst / max(qmax, 1e-300)
    print("  %s max|dM/dt+Q_out|=%.4e  /max Q_out=%.3f （判据 <5e-2）"
          % (tag, worst, ratio))
    return ratio


# ============================================================================
# 10. 评估与出图（B1 修复：统一 .cpu()）
# ============================================================================
def snap_times():
    return [round(0.1 * k, 1) for k in range(int(round(T_END / 0.1)) + 1)]


def evaluate_snapshots(net):
    ts = snap_times()
    x = torch.linspace(X0, X1, N_SNAP_GRID, dtype=torch.float64, device=DEVICE)
    eta_s, u1_s, zeta_s = [], [], []
    for tv in ts:
        Xc = torch.stack([x, torch.full_like(x, tv)], dim=1)
        eta, u1 = net(Xc)
        eta_s.append(eta.detach().cpu().numpy())
        u1_s.append(u1.detach().cpu().numpy())
        zeta_s.append((bed_z(x) + gam(x) * eta).detach().cpu().numpy())
    xnp = x.detach().cpu().numpy()
    Anp = cap_A(x).detach().cpu().numpy()
    mass = [float(np.sum(Anp * e * (X1 - X0) / (N_SNAP_GRID - 1))) for e in eta_s]
    q_out = []
    for tv in ts:
        Xc = torch.stack([torch.tensor([X1], dtype=torch.float64, device=DEVICE),
                          torch.tensor([tv], dtype=torch.float64, device=DEVICE)],
                         dim=1)
        eta, u1 = net(Xc)
        q_out.append(float((cap_A(Xc[:, 0]) * eta * u1).reshape(()).detach()))
    return (np.array(ts), xnp, np.array(eta_s), np.array(u1_s),
            np.array(zeta_s), np.array(mass), np.array(q_out))


def rel_l1_vs_ref(net, ref_path, outdir):
    """论文口径：rel-L1(t) = Σ|η_pinn−η_ref|/Σ|η_ref| + 质量对照（同一 1D 积分）。"""
    rd = np.load(ref_path)
    xr, tr, er = rd["x"], rd["t"], rd["eta"]
    ts, xnp, eta_s, _, _, mass, _ = evaluate_snapshots(net)
    xrn = np.asarray(xr, float)
    Anp = np.sqrt(1.0 + (BED_A * (xrn - X1)) ** 2)
    lines = ["t  relL1  M_pinn  M_ref"]
    out = []
    for tv, e, mp in zip(ts, eta_s, mass):
        k = int(np.argmin(np.abs(np.asarray(tr, float) - tv)))
        er_k = np.asarray(er[k], float)
        ep = np.interp(xrn, xnp, e)
        denom = np.sum(np.abs(er_k))
        rel = float(np.sum(np.abs(ep - er_k)) / max(denom, 1e-300))
        m_ref = float(np.sum(Anp * er_k) * (xrn[1] - xrn[0]))
        lines.append("%.1f  %.4e  %.6f  %.6f" % (tv, rel, mp, m_ref))
        out.append((tv, rel))
    with open(os.path.join(outdir, "rel_l1_vs_t.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    tv_f, rel_f = out[-1]
    print("  [rel-L1] 终态 t=%.1f: rel-L1=%.4e （判据 5e-2） | 明细 rel_l1_vs_t.txt"
          % (tv_f, rel_f))
    return rel_f


def make_plots(outdir, ts, x, eta_s, u1_s, mass, q_out, hist, ref_path=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(outdir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    it = np.asarray(hist["it"], float)
    for k in ("total", "pde", "ic", "bc", "cons", "pos"):
        if hist.get(k) and len(hist[k]) > 0:
            ax.semilogy(it, np.asarray(hist[k], float) + 1e-300, label=k)
    ax.set_xlabel("iteration"); ax.set_ylabel("loss"); ax.legend(fontsize=8)
    ax.set_title("1D intrinsic PINN v2 (%s): loss history" % MODE)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "pinn1d_convergence.png"),
                                    dpi=150)
    plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    axs[0].plot(ts, q_out, "-o", ms=3, color="#C0392B", label="PINN 1D v2")
    axs[0].set_xlabel("$t$ [s]"); axs[0].set_ylabel("$Q_{out}$ [m$^2$/s]")
    axs[0].legend()
    axs[1].plot(ts, mass, "-o", ms=3, color="#185FA5", label="PINN 1D v2")
    axs[1].set_xlabel("$t$ [s]"); axs[1].set_ylabel("$M(t)$ [m$^2$]"); axs[1].legend()
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "pinn1d_discharge_mass.png"),
                                    dpi=150)
    plt.close(fig)

    if ref_path and os.path.exists(ref_path):
        rd = np.load(ref_path)
        xr, tr, er = rd["x"], rd["t"], rd["eta"]
    else:
        xr = tr = er = None
    n = len(ts); ncol = 4; nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(1.75 * ncol, 1.30 * nrow),
                             sharex=True, sharey=True)
    for j, tv in enumerate(ts):
        ax = axes.flat[j]
        ax.plot(x, eta_s[j], color="#C0392B", lw=1.4)
        if er is not None:
            k = int(np.argmin(np.abs(np.asarray(tr, float) - tv)))
            ax.plot(np.asarray(xr, float), np.asarray(er[k], float),
                    color="#185FA5", lw=1.2, ls="--")
        ax.axvline(2.0, color="0.6", ls=":", lw=0.7)
        ax.set_title("$t=%g$ s" % tv, fontsize=10, pad=2)
        ax.tick_params(labelsize=8)
    for r in range(nrow):
        for cc in range(ncol):
            axes[r][cc].set_ylim(-0.02, 1.05)
    fig.suptitle("1D intrinsic PINN v2-%s (red) vs reference (blue dashed)" % MODE,
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(outdir, "pinn1d_snap_grid.png"), dpi=150)
    plt.close(fig)
    print("  图件已写出（convergence / discharge_mass / snap_grid）")


# ============================================================================
# 11. 训练（v2：Adam-only 默认；slabfv 两遍法逐块反传）
# ============================================================================
def train(net, Xp, Xi_main, Xi_band, Xb0, Xb1, outdir):
    hist = {"it": [], "total": [], "pde": [], "ic": [], "bc": [], "cons": [],
            "pos": [], "out": []}
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.StepLR(optimizer,
                                            step_size=max(N_ITER // 2, 1),
                                            gamma=1e-1)
    t_start = time.time()

    def log_step(it, L, Lp, Lic, Lbc, Lc, Lpos):
        el = time.time() - t_start
        print("  step %6d | L %.4e | pde %.4e | ic %.4e | bc %.4e | "
              "cons/slab %.4e | pos %.4e | %.0fs"
              % (it, L, Lp, Lic, Lbc, Lc, Lpos, el))

    print("  ---- Adam %d 步 （mode=%s, causal=%s） ----" % (N_ITER, MODE, CAUSAL))
    for it in range(1, N_ITER + 1):
        optimizer.zero_grad()
        if MODE == "slabfv":
            # pass1: no_grad 定权（S1 两遍法）
            if CAUSAL:
                with torch.no_grad():
                    Ls = slab_losses(net, with_grad=False)
                w = _causal_weights(Ls, _eps_causal(it))
            else:
                w = torch.ones(N_SLAB, dtype=torch.float64, device=DEVICE)
            # 点态/IC/BC 部分
            L0, Lp, Lic, Lbc, _, Lpos, Lout = total_loss(net, Xp, Xi_main,
                                                         Xi_band, Xb0, Xb1, it)
            L0.backward()
            # pass2: 逐块重评估 + 立即反传（每块独立图 ⇒ 显存有界）
            (xe, dx, xg, tb, dt, tau) = _slab_grid()
            m0, M0 = _ic_cell_means()
            L_slab_acc = torch.zeros((), dtype=torch.float64, device=DEVICE)
            for s0 in range(0, N_SLAB, SLAB_CHUNK):
                grp = list(range(s0, min(s0 + SLAB_CHUNK, N_SLAB)))
                lv = grp + [grp[-1] + 1]
                mbar, Mbar = slab_cell_means(net, levels=lv)
                if s0 == 0:
                    mbar = torch.cat([m0.unsqueeze(0), mbar[1:]], dim=0)
                    Mbar = torch.cat([M0.unsqueeze(0), Mbar[1:]], dim=0)
                Q_e, F_e = slab_edge_fluxes(net, slabs=grp)
                res_m, res_p = [], []
                for k, i in enumerate(grp):
                    wqt_i = _SLAB_CACHE["wqt"].reshape(N_SLAB, NQ_T)[i]
                    src = slab_source_ints(net, [i]).squeeze(0)
                    rm, rp = _assemble_slab(mbar[k + 1], Mbar[k + 1], Q_e[k],
                                            F_e[k], src, i, wqt_i, dx)
                    rm = rm - mbar[k]
                    rp = rp - Mbar[k]
                    res_m.append(rm)
                    res_p.append(rp)
                rm = torch.stack(res_m)
                rp = torch.stack(res_p)
                Lc = torch.mean((rm / S_M) ** 2 * W_SLAB_M
                                + (rp / S_P) ** 2 * W_SLAB_P, dim=1)
                wc = w[torch.tensor(grp, device=DEVICE)]
                loss_chunk = torch.sum(wc * Lc) / N_SLAB
                loss_chunk.backward(retain_graph=False)
                L_slab_acc = L_slab_acc + loss_chunk.detach()
            Ltot = float(L0.detach()) + float(L_slab_acc)
            Lc_val = float(L_slab_acc) * N_SLAB     # 反归一记录值
            if it % LOG_EVERY == 0 or it == 1:
                log_step(it, Ltot, float(Lp), float(Lic), float(Lbc), Lc_val,
                         float(Lpos))
            if it % RECORD_EVERY == 0 or it == 1:
                hist["it"].append(it)
                hist["total"].append(Ltot)
                hist["pde"].append(float(Lp)); hist["ic"].append(float(Lic))
                hist["bc"].append(float(Lbc)); hist["cons"].append(Lc_val)
                hist["pos"].append(float(Lpos)); hist["out"].append(float(Lout))
        else:
            L, Lp, Lic, Lbc, Lc, Lpos, Lout = total_loss(net, Xp, Xi_main,
                                                         Xi_band, Xb0, Xb1, it)
            L.backward()
            if it % LOG_EVERY == 0 or it == 1:
                log_step(it, float(L), float(Lp), float(Lic), float(Lbc),
                         float(Lc), float(Lpos))
            if it % RECORD_EVERY == 0 or it == 1:
                hist["it"].append(it)
                for k, v in (("total", L), ("pde", Lp), ("ic", Lic), ("bc", Lbc),
                             ("cons", Lc), ("pos", Lpos), ("out", Lout)):
                    hist[k].append(float(v))
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()
        sched.step()

    if USE_LBFGS:
        if MODE == "slabfv":
            raise RuntimeError("L-BFGS 与 slabfv 两遍法不兼容（目标须逐步一致）")
        print("  ---- L-BFGS-B %d 步 ----" % LBFGS_MAXITER)
        from scipy.optimize import minimize as scipy_minimize
        params = list(net.parameters())
        shapes = [q.shape for q in params]
        sizes = [int(np.prod(s)) for s in shapes]
        offsets = np.cumsum([0] + sizes)

        def flatten(ps):
            return torch.cat([q.reshape(-1) for q in ps]).detach().cpu().numpy() \
                .astype(np.float64)

        def unflatten(flat):
            ft = torch.tensor(flat, dtype=torch.float64, device=DEVICE)
            with torch.no_grad():
                for i, q in enumerate(params):
                    q.copy_(ft[offsets[i]:offsets[i + 1]].reshape(shapes[i]))

        def value_and_grad(flat_np):
            unflatten(flat_np)
            for q in params:
                if q.grad is not None:
                    q.grad = None
            L, Lp, Lic, Lbc, Lc, Lpos, Lout = total_loss(net, Xp, Xi_main,
                                                         Xi_band, Xb0, Xb1,
                                                         it=N_ITER)
            L.backward()
            gf = torch.cat([(q.grad if q.grad is not None
                             else torch.zeros_like(q)).reshape(-1)
                            for q in params]).detach().cpu().numpy() \
                .astype(np.float64)
            return float(L.detach()), gf

        res = scipy_minimize(value_and_grad, flatten(params), method="L-BFGS-B",
                             jac=True,
                             options={"maxiter": LBFGS_MAXITER,
                                      "maxfun": LBFGS_MAXFUN,
                                      "ftol": 1e-12, "gtol": 1e-12})
        unflatten(res.x)
        print("  L-BFGS-B 完成: success=%s, iterations=%d, objective=%.4e"
              % (bool(res.success), int(res.nit), float(res.fun)))
    return hist


# ============================================================================
# 12. main
# ============================================================================
def main():
    global N_INT, N_IC, N_BC, N_ITER, SEED, N_DENSE_AUDIT, RESIDUAL_ASSEMBLY
    global DEVICE, MODE, EPS_VISC, N_SLAB, N_CELL, NQ_X, NQ_T, CAUSAL, USE_LBFGS
    global W_SLAB_M, W_SLAB_P, SLAB_CHUNK

    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--mode", default="adam", choices=["adam", "viscous", "slabfv"])
    ap.add_argument("--n-int", type=int, default=None)
    ap.add_argument("--n-ic", type=int, default=None)
    ap.add_argument("--n-bc", type=int, default=None)
    ap.add_argument("--iter", type=int, default=None)
    ap.add_argument("--n-dense-audit", type=int, default=None)
    ap.add_argument("--assembly", default=None, choices=["hybrid", "flux"])
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--eps-visc", type=float, default=None)
    ap.add_argument("--n-slab", type=int, default=None)
    ap.add_argument("--n-cell", type=int, default=None)
    ap.add_argument("--nq-x", type=int, default=None)
    ap.add_argument("--nq-t", type=int, default=None)
    ap.add_argument("--w-slab-m", type=float, default=None)
    ap.add_argument("--w-slab-p", type=float, default=None)
    ap.add_argument("--slab-chunk", type=int, default=None)
    ap.add_argument("--no-causal", action="store_true")
    ap.add_argument("--lbfgs", action="store_true")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--ref", default=None, help="参考解 npz（rel-L1/叠加图用）")
    ap.add_argument("--outdir", default="output/pinn1d_v2")
    a = ap.parse_args()

    if a.n_int: N_INT = a.n_int
    if a.n_ic: N_IC = a.n_ic
    if a.n_bc: N_BC = a.n_bc
    if a.iter: N_ITER = a.iter
    if a.n_dense_audit: N_DENSE_AUDIT = a.n_dense_audit
    if a.assembly: RESIDUAL_ASSEMBLY = a.assembly
    if a.seed: SEED = a.seed
    MODE = a.mode
    if a.eps_visc is not None: EPS_VISC = a.eps_visc
    if a.n_slab is not None: N_SLAB = a.n_slab
    if a.n_cell is not None: N_CELL = a.n_cell
    if a.nq_x is not None: NQ_X = a.nq_x
    if a.nq_t is not None: NQ_T = a.nq_t
    if a.w_slab_m is not None: W_SLAB_M = a.w_slab_m
    if a.w_slab_p is not None: W_SLAB_P = a.w_slab_p
    if a.slab_chunk is not None: SLAB_CHUNK = a.slab_chunk
    CAUSAL = not a.no_causal
    USE_LBFGS = a.lbfgs
    DEVICE = torch.device(a.device)
    if a.device == "cuda":
        torch.set_default_device("cuda")
    if MODE == "viscous":
        RESIDUAL_ASSEMBLY = "flux"     # 黏性定义在守恒变量上

    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), a.outdir) \
        if not os.path.isabs(a.outdir) else a.outdir
    os.makedirs(outdir, exist_ok=True)

    print("=" * 78)
    print("1D 内蕴 PINN v2（E3 抛物床） mode=%s" % MODE)
    print("=" * 78)
    print("域 [%.1f,%.1f] × [0,%.1f]  床 z=(x¹−10)²/25  坝 x¹=%.1f"
          % (X0, X1, T_END, DAM_X))
    print("网络: %d×%d N-LAAF n=%g float64 device=%s"
          % (LAYERS, WIDTH, N_SCALE, DEVICE))
    if MODE == "viscous":
        print("黏性: ε=%g（守恒变量 m=Aη, M=Aηu¹）" % EPS_VISC)
    if MODE == "slabfv":
        print("slab-FV: %d cell × %d slab, NQ_X=%d NQ_T=%d, W_M=%g W_P=%g"
              % (N_CELL, N_SLAB, NQ_X, NQ_T, W_SLAB_M, W_SLAB_P))
        print("        S_M=%g S_P=%g, 旧全局守恒罚已移除（防双计）" % (S_M, S_P))
    print("因果加权: %s（ε_c=%g 起，每 1/3 步 ×10）" % (CAUSAL, EPS_CAUSAL0))
    print("配点: N_INT=%d, N_IC=%d(+%d%%), N_BC=%d/边"
          % (N_INT, N_IC, 100 * IC_BAND_FRAC, N_BC))
    print("训练: Adam %d（L-BFGS %s）" % (N_ITER, "开" if USE_LBFGS else "关"))

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    with torch.device("cpu"):
        net = NLAFNet()
    net = net.to(DEVICE, dtype=torch.float64)
    for p in net.parameters():
        p.requires_grad_(True)
    net.double()

    Xp, Xi_main, Xi_band, Xb0, Xb1 = make_points()
    Xp = Xp.to(DEVICE).requires_grad_(True)
    Xi_main = Xi_main.to(DEVICE)
    Xi_band = Xi_band.to(DEVICE)
    Xb0 = Xb0.to(DEVICE)
    Xb1 = Xb1.to(DEVICE)

    print()
    print("=" * 78)
    print("  训练前审计")
    print("=" * 78)
    ok = True
    ok &= audit_geometry()
    audit_ic_mass()
    ok &= audit_static_lake()
    ok &= audit_pressure_split()
    Xchk = dense_audit_points(2048).to(DEVICE).requires_grad_(True)
    ok &= audit_assembly_identity(net, Xchk)
    if MODE == "slabfv":
        ok &= audit_slab_static_lake()
        ok &= audit_slab_manufactured()
    if not ok:
        print("  训练前审计未全过，中止。")
        sys.exit(1)

    print()
    print("=" * 78)
    print("  训练")
    print("=" * 78)
    hist = train(net, Xp, Xi_main, Xi_band, Xb0, Xb1, outdir)

    # B2：训练一结束立刻存权重
    sd_path = os.path.join(outdir, "net_state.pt")
    torch.save({k: v.detach().cpu() for k, v in net.state_dict().items()}, sd_path)
    print("  state_dict 已存: %s" % sd_path)

    print()
    print("=" * 78)
    print("  训练后审计")
    print("=" * 78)
    sampling_adequacy_report(net, Xp, tag="训练后")
    mass_balance_report(net, tag="训练后")
    if MODE == "slabfv":
        with torch.no_grad():
            Ls = slab_losses(net, with_grad=False)
        print("  [训练后 slab] mean L=%.4e  max L=%.4e"
              % (float(torch.stack(Ls).mean()), float(torch.stack(Ls).max())))

    ts, x, eta_s, u1_s, zeta_s, mass, q_out = evaluate_snapshots(net)
    npz_path = os.path.join(outdir, "pinn1d_dam_parab.npz")
    np.savez(npz_path,
             x=x, t_list=np.array(ts), prof_h=eta_s, prof_u1=u1_s,
             prof_zeta=zeta_s, mass=np.array(mass), q_out=np.array(q_out),
             hist_it=np.array(hist["it"]),
             hist_total=np.array(hist["total"]),
             hist_pde=np.array(hist["pde"]),
             hist_ic=np.array(hist["ic"]),
             hist_bc=np.array(hist["bc"]),
             hist_cons=np.array(hist["cons"]),
             hist_pos=np.array(hist["pos"]),
             cfg=np.array([X0, X1, T_END, DAM_X, H_UP, H_DN, N_SCALE, W_A,
                           W_I, W_IC_BAND, W_B, W_CONS, W_POS]))
    print("  npz 已存: %s" % npz_path)

    with open(os.path.join(outdir, "error_vs_iteration.txt"), "w",
              encoding="utf-8") as f:
        f.write("iter total pde ic bc cons pos\n")
        for i, itv in enumerate(hist["it"]):
            f.write("%d %.6e %.6e %.6e %.6e %.6e %.6e\n"
                    % (itv, hist["total"][i], hist["pde"][i], hist["ic"][i],
                       hist["bc"][i], hist["cons"][i], hist["pos"][i]))

    if a.ref:
        try:
            rel_l1_vs_ref(net, a.ref, outdir)
        except Exception as exc:
            print("  [rel-L1] 参考解对照失败: %r" % (exc,))

    gamX = float(gam(torch.tensor([X1], device=DEVICE)))
    lines = ["t  eta(10)  u1(10)  Fr"]
    for tv, e, u in zip(ts, eta_s, u1_s):
        fr = abs(u[-1]) / math.sqrt(G_GRAV * max(e[-1], 1e-12) * gamX)
        lines.append("%.1f  %.6f  %.6f  %.4f" % (tv, e[-1], u[-1], fr))
    with open(os.path.join(outdir, "bc_probe.log"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("  出口探针已写 bc_probe.log（末行: %s）" % lines[-1])

    if not a.no_plot:
        make_plots(outdir, ts, x, eta_s, u1_s, mass, q_out, hist, ref_path=a.ref)

    print("=" * 78)
    print("  全部完成。")
    print("=" * 78)


if __name__ == "__main__":
    main()
