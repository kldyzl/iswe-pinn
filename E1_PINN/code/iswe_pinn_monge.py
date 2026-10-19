# -*- coding: utf-8 -*-
r"""
iswe_pinn_monge.py — ISWE-PINN（Monge 度量 / 非正交坐标基 版）
=================================================================
本文件是 E1（双周期床良平衡实验）的求解器，床面 B(x,y)=0.3 sin(πx)cos(πy)
使 z_1 z_2 ≢ 0，故必须在**非正交 Monge 度量**下求解；
正交版（Bachini & Putti 的 Gram–Schmidt 标架写法）在此地形上不适用。

数学依据（主：论文 Geometrically intrinsic modeling of shallow water flows,
Bachini & Putti, ESAIM: M2AN 54 (2020)；辅：本项目的 Monge 推导文档
ISWE_Monge_derivation.tex，已过九轮独立审核）:
  切向坐标基   b_i = ∂_i X = (δ_{i1}, δ_{i2}, z_i)          (eq:evec)
  诱导度量     G_ij = b_i·b_j = δ_ij + z_i z_j              (eq:G)
  行列式       D    = 1 + |∇z|²                              (eq:detG)
  逆度量       G^{ij}= δ^{ij} − z_i z_j / D                  (eq:Ginv)
  几何因子     γ    = 1/√D,   J = √det G = √D = 1/γ          (eq:gamma, eq:J)
  联络         Γ^k_ij = z_k z_ij / D                          (eq:ChrisClosed)
  张量散度     (div T)^j = (1/J)∂_i(J T^{ij}) + Γ^j_ik T^{ik} (eq:divT_comp)
  通量         F^{ij} = q^i q^j/η + (gη²/2) γ G^{ij}         (eq:U)
  源项         S^j = (gη²/2) G^{ji}∂_i γ + gη G^{ji}∂_i z     (eq:S)
  良平衡       γη + z = const ⇔ η = (ζ0 − z)/γ               (prop:rest)

与正交版的唯一结构性差别:
  - G^{12} = −z_1 z_2 / D ≢ 0 ⇒ 压力张量有**非对角分量**（正交版为 0）；
  - 联络换成 Γ^k_ij = z_k z_ij / D（取代尺度因子 Christoffel）；
  - 源项 S^j 保留 G^{ji} 的**完整收缩（含交叉项）**。
  当 z_1 z_2 ≡ 0（E2 平底 / E3 一维）时两版逐元重合。
实测（_audit_monge_geometry_np.py，29/29 PASS；服务器 torch 复跑同）:
  - max|G^{12}| = 0.153（正交版 0）；max|F^{12}| = 0.404；S^1 交叉项 0.578
  - 静水残差 max|R2| ≈ 2.7e-15（与正交版同量级 ⇒ 该判据与度量无关）
  - 度量相容性 (1/J)∂_i(JG^{ij}) + Γ^j_ik G^{ik} ≈ 2.4e-10
  - 训练前/后良平衡审计均"通过 ✅"

结构 / 输入输出（与早期单体实现逐模块对应；后端由 TF 换 PyTorch）:
  1. 全局常量（G_GRAV=9.81, ZETA0=0.5, T_END=0.5, LX=LY=2.0）
  2. geometry_monge(x, y) -> 17 个几何量（序号/含义与旧正交版一一对应）
  3. SWENet(width=128, layers=6)，输入 (x,y,t)→归一化[-1,1]，
     输出层 he_normal + 常数偏置 INIT_BIAS，扰动参数化 η = η_r + δ
  4. flux_and_source(eta, q1, q2, geo)      （含 G^{ji} 完整收缩）
  5. residuals(X, net) -> R1, R2, R3, eta, geo
  6. total_loss = L_pde + W_FS·L_fs + W_IBC·(L_i + L_b)
  7. make_points(nx=30, ny=30, nt=10, m=30)
  8. main(): Adam(1e-3, ExpDecay 800/0.8, clip 1.0, N_ITER=3000)
     → scipy L-BFGS-B(maxiter=30000, maxfun=100000, ftol=gtol=1e-12)
  9. 评估 ne=120 网格 @ t=T_END + 三张图

产物（均落在 --outdir，默认 = 本脚本所在目录，即代码文件夹）:
  iswe_pinn_result_monge.npz     数值解（zeta/eta/B/u1/u2 + 误差历史）
  iswe_pinn_monge_errors.txt     逐次迭代误差日志（Adam/L-BFGS 每次一行）
  error_vs_iteration.txt         按 RECORD_EVERY 间隔的误差表（画图用）
  pinn_free_surface_monge.png    图1 t=T_END 预测自由水面 ζ
  pinn_convergence_monge.png     图2 误差随迭代变化（四曲线）
  pinn_topography_monge.png      图3 床面地形 B(x,y)

运行:
  python iswe_pinn_monge.py [--fast] [--no-plot] [--outdir DIR]
  --fast 仅用于快速冒烟测试（缩配点/网络/迭代；默认参数=E1 正式配置）
"""
import os
import argparse
import numpy as np
import torch

torch.set_default_dtype(torch.float64)


# ---- 计算设备（严格审核 P0：原版无任何 device 迁移，全在 CPU 上跑）----
def resolve_device(spec="auto"):
    """auto: 有可用 CUDA 就用 cuda，否则 cpu。可显式指定 cpu / cuda / cuda:N。"""
    if spec in (None, "", "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


DEVICE = resolve_device("auto")   # 可被 --device 覆盖（见 main）

# ==================== 全局物理与计算参数（与 pinn_swe_combined2.py 一致）====
G_GRAV = 9.81          # 重力加速度 (代码中用 G_GRAV 避免与度量行列式 g 混淆)
ZETA0  = 0.5           # 静水自由水面高程
T_END  = 0.5           # 时间终点 (秒)
LX, LY = 2.0, 2.0      # 计算域

# ---- 初始条件: 纯静水态 (lake-at-rest) ----
# 目标: η = η0 = (ζ0 - B)/N, q1 = q2 = 0；网络输出扰动 δ
# INIT_BIAS=0.25 ⇒ 从非精确初值出发 (δ≈0.25, q≈0.25)，检验向真解收敛

# ---- 训练参数 ----
N_ITER       = 3000    # Adam 阶段迭代次数
LOG_EVERY    = 5       # 控制台打印间隔
RECORD_EVERY = 20      # 误差记录间隔

# ---- 输出层初始偏置（与 combined2 相同）----
INIT_BIAS = 0.25

# ---- 损失权重（与 combined2 相同）----
W_FS  = 1.0            # 自由水面约束: 显式钉住 δ=0 (ζ=ζ0)
W_IBC = 10.0            # 初边值条件权重

# ---- 配点（与 combined2 的 make_points 默认相同）----
NX, NY, NT, M_BC = 30, 30, 10, 30

# ---- 网络规模（与 combined2 的 SWENet 默认相同）----
WIDTH, LAYERS = 128, 6

# ---- L-BFGS 上限（与 combined2 的 scipy options 相同）----
LBFGS_MAXITER = 30000
LBFGS_MAXFUN  = 100000

SEED = 42


# ================================================================
# 1. 解析几何量 —— Monge 度量（eq:G/Ginv/ChrisClosed + divT_comp）
#    返回与旧正交版完全相同序号/含义的 17 个几何量:
#      B, Bx, By, sg, N, N1, N2,
#      g11, g22, g12(=G^{12}≠0),
#      G1_11, G1_12, G1_22, G2_11, G2_12, G2_22
# ================================================================
def geometry_monge(x, y):
    """
    地形: B(x,y) = 0.3 sin(πx) cos(πy)（与正交版/combined2 完全一致，不改）

    Monge 坐标下的床面几何（推导文档 ISWE_Monge_derivation.tex）:
      切向基      b_i = ∂_i X = (δ_{i1}, δ_{i2}, z_i)        (eq:evec)
      诱导度量    G_ij = b_i·b_j = δ_ij + z_i z_j            (eq:G)
      行列式      D    = 1 + |∇z|²                            (eq:detG)
      逆度量      G^{ij}= δ^{ij} − z_i z_j / D                (eq:Ginv)
      法向/几何因子 γ  = N·e3 = 1/√D                          (eq:gamma)
      面积元系数  J    = √det G = √D = 1/γ                     (eq:J)
      联络        Γ^k_ij = z_k z_ij / D                        (prop:chris, eq:ChrisClosed)

    与正交版的关系（数值恒等项）:
      sg = h1·h2 = √(1+|∇B|²) = √D  →  与正交版逐位相同
      N  = 1/sg = 1/√D = γ          →  与正交版逐位相同
      N1, N2 = ∂_i γ = −γ³ (B_k B_ki) → 与正交版同式同值
    真正的差别只在:
      G^{12} = −z1 z2 / D ≠ 0   （非对角压力项出现）
      Γ^k_ij = z_k z_ij / D     （取代尺度因子 Christoffel）

    返回 17 个几何量，序号/含义与正交版严格一一对应:
      B, Bx, By, sg, N, N1, N2,
      g11, g22, g12,
      G1_11, G1_12, G1_22, G2_11, G2_12, G2_22
    """
    # ---- 地形（部分不改）----
    B   =  0.3 * torch.sin(torch.pi * x) * torch.cos(torch.pi * y)
    Bx  =  0.3 * torch.pi * torch.cos(torch.pi * x) * torch.cos(torch.pi * y)
    By  = -0.3 * torch.pi * torch.sin(torch.pi * x) * torch.sin(torch.pi * y)
    Bxx = -0.3 * torch.pi**2 * torch.sin(torch.pi * x) * torch.cos(torch.pi * y)
    Byy =  Bxx
    Bxy = -0.3 * torch.pi**2 * torch.cos(torch.pi * x) * torch.sin(torch.pi * y)

    # ---- 度量行列式与几何因子（eq:detG / eq:gamma / eq:J）----
    D = 1.0 + Bx**2 + By**2          # det G = 1 + |∇B|²
    sg = torch.sqrt(D)               # J = √D = h1·h2（与正交版逐位相同）
    N  = 1.0 / sg                    # γ = 1/√D（与正交版逐位相同）

    # ---- ∂_i γ = −γ³ (B_k B_ki)（与正交版同式）----
    N1 = -(N**3) * (Bx * Bxx + By * Bxy)   # ∂_1 γ
    N2 = -(N**3) * (Bx * Bxy + By * Byy)   # ∂_2 γ

    # ---- 逆度量 G^{ij} = δ^{ij} − z_i z_j / D（eq:Ginv）----
    g11 = 1.0 - Bx**2 / D            # G^{11}
    g22 = 1.0 - By**2 / D            # G^{22}
    G12 = -Bx * By / D               # G^{12} = −z1 z2 / D  ← 非对角，正交版为 0
    g12 = G12

    # ---- 联络 Γ^k_ij = z_k z_ij / D（eq:ChrisClosed）----
    G1_11 = Bx * Bxx / D             # Γ^1_11
    G1_12 = Bx * Bxy / D             # Γ^1_12
    G1_22 = Bx * Byy / D             # Γ^1_22
    G2_11 = By * Bxx / D             # Γ^2_11
    G2_12 = By * Bxy / D             # Γ^2_12
    G2_22 = By * Byy / D             # Γ^2_22

    return (B, Bx, By, sg, N, N1, N2,
            g11, g22, g12,
            G1_11, G1_12, G1_22,
            G2_11, G2_12, G2_22)


# 兼容别名：旧审计脚本可能按名调用 geometry_orthogonal（返回 17 元，含义相同）。
geometry_orthogonal = geometry_monge

# ================================================================
# 2. 神经网络: 输入 (x,y,t) → 输出 (δ, q1, q2), 物理水深 η = η_r + δ
#    （与早期单体实现的 SWENet 逐层对应，TF Dense → torch Linear）
# ================================================================
class SWENet(torch.nn.Module):
    def __init__(self, width=128, layers=6):
        super().__init__()
        # 隐藏层: Dense(width, tanh, he_normal) × layers
        self.hids = torch.nn.ModuleList()
        d_in = 3
        for _ in range(layers):
            lin = torch.nn.Linear(d_in, width)
            # keras he_normal: std=√(2/fan_in) ⇒ 增益 √2 ⇒ torch nonlinearity="relu"
            # （tanh 增益 5/3≈1.667 与 he_normal 不符，差约 18%，已修正）
            torch.nn.init.kaiming_normal_(lin.weight, nonlinearity="relu")
            torch.nn.init.zeros_(lin.bias)                                   # keras 默认 bias=0
            self.hids.append(lin)
            d_in = width
        # 输出层: Dense(3, bias = INIT_BIAS)（keras 默认 kernel=glorot_uniform）
        self.out = torch.nn.Linear(width, 3)
        torch.nn.init.xavier_uniform_(self.out.weight)
        torch.nn.init.constant_(self.out.bias, INIT_BIAS)

    def forward(self, X):
        # 输入归一化到 [-1, 1]
        h = torch.stack([
            X[:, 0] / LX * 2 - 1,
            X[:, 1] / LY * 2 - 1,
            X[:, 2] / T_END * 2 - 1
        ], dim=1)
        for layer in self.hids:
            h = torch.tanh(layer(h))
        r = self.out(h)

        # 静水参考态 η_r = (ζ0 - B)/N —— 解析已知, 逐点计算 (自动微分沿此传播)
        x, y = X[:, 0], X[:, 1]
        geo = geometry_monge(x, y)
        B, N = geo[0], geo[4]
        eta_ref = (ZETA0 - B) / N

        # 扰动参数化: 网络只学偏离量 δ, 物理水深 = 参考态 + 偏离
        delta_eta = r[:, 0]
        q1 = r[:, 1]
        q2 = r[:, 2]

        return eta_ref + delta_eta, q1, q2   # η, q^1, q^2


# ================================================================
# 3. 通量 + Christoffel 收缩 + 几何源项（论文 Eq.(3.2)/(3.3)，无粘性）
# ================================================================
def flux_and_source(eta, q1, q2, geo):
    """
    依据 Monge 度量下的 ISWE（eq:U / eq:S；g^{12}≠0，非对角）:
      F^{ij} = q^i q^j / η + p G^{ij},   p = (g η²/2) γ,  γ = ∂x³/∂s³ = 1/√D
      S^j   = (gη²/2) G^{ji} ∂_i γ + gη G^{ji} ∂_i z     （完整收缩，含交叉项）
      Ch^j  = Γ^j_ik F^{ik}   （eq:divT_comp 协变散度的 Christoffel 项）
    无粘性: T_sw = 0, f_B = 0。
    """
    (B, Bx, By, sg, N, N1, N2,
     g11, g22, g12,
     G1_11, G1_12, G1_22,
     G2_11, G2_12, G2_22) = geo

    h_safe = torch.clamp(eta, min=1e-8)     # 防除零（与 combined2 的 1e-8 相同）
    u1 = q1 / h_safe
    u2 = q2 / h_safe

    p = 0.5 * G_GRAV * eta**2 * N           # 压力标量 p = (gη²/2)·∂x³/∂s³

    # 通量 F^{ij} = q^i u^j + p G^{ij}（Monge 度量：G^{12} ≠ 0，压力有非对角分量）
    F11 = q1 * u1 + p * g11
    F12 = q1 * u2 + p * g12                 # = q1 q2/η + p·G^{12}（Eq.(U) 非对角元）
    F22 = q2 * u2 + p * g22

    # Christoffel 收缩 (Γ^i_12 = Γ^i_21, 交叉项系数为 2)
    Ch1 = G1_11 * F11 + 2.0 * G1_12 * F12 + G1_22 * F22
    Ch2 = G2_11 * F11 + 2.0 * G2_12 * F12 + G2_22 * F22

    # 几何源项 S^j = (gη²/2) G^{ji} ∂_i γ + gη G^{ji} ∂_i z   (eq:S)
    # Monge 度量 g^{12}≠0 → 必须保留完整收缩（含交叉项）
    S1 = (0.5 * G_GRAV * eta**2 * (g11 * N1 + g12 * N2)
          + G_GRAV * eta * (g11 * Bx + g12 * By))
    S2 = (0.5 * G_GRAV * eta**2 * (g12 * N1 + g22 * N2)
          + G_GRAV * eta * (g12 * Bx + g22 * By))

    return F11, F12, F22, Ch1, Ch2, S1, S2


# 兼容别名：audit_cross.py 双实现交叉审计按旧符号名调用本模块的正交通量/源项装配。
# geometry_orthogonal 重写时已保留同名；仅 flux_and_source 由旧名
# flux_and_source_orthogonal 改来，故此处提供别名以兼容既有审计脚本。
flux_and_source_orthogonal = flux_and_source


# ================================================================
# 4. PDE 残差（完整 ISWE，Monge 度量展开 eq:divT_comp/eq:U/eq:S，无粘性）
# ================================================================
def residuals(X, net):
    """
    质量方程:  R1 = ∂t η + (1/√g)[∂₁(√g q¹) + ∂₂(√g q²)]
    s¹-动量:   R2 = ∂t q¹ + (1/√g)[∂₁(√g F¹¹) + ∂₂(√g F¹²)] + Ch¹ + S¹
    s²-动量:   R3 = ∂t q² + (1/√g)[∂₁(√g F¹²) + ∂₂(√g F²²)] + Ch² + S²
    （(1/J)∂_i(J·) + Γ 收缩 ≡ eq:divT_comp 的协变散度，已符号验证）

    返回: R1, R2, R3, eta, geo
    """
    X = X.clone().requires_grad_(True)
    eta, q1, q2 = net(X)
    geo = geometry_monge(X[:, 0], X[:, 1])
    sg = geo[3]

    F11, F12, F22, Ch1, Ch2, S1, S2 = flux_and_source(eta, q1, q2, geo)

    # √g 加权量（协变散度的标准形式）
    sg_eta = sg * eta
    sg_q1  = sg * q1
    sg_q2  = sg * q2
    sg_F11 = sg * F11
    sg_F12 = sg * F12
    sg_F22 = sg * F22

    # 一次性取所有梯度（grad 分量: [0]=∂₁, [1]=∂₂, [2]=∂t）
    def grad_of(v):
        return torch.autograd.grad(v.sum(), X, create_graph=True)[0]

    g_eta = grad_of(sg_eta)
    g_q1  = grad_of(sg_q1)
    g_q2  = grad_of(sg_q2)
    g_F11 = grad_of(sg_F11)
    g_F12 = grad_of(sg_F12)
    g_F22 = grad_of(sg_F22)

    # 质量方程: (1/√g)∂t(√g η) = ∂t η（√g 与 t 无关）
    R1 = (g_eta[:, 2] + g_q1[:, 0] + g_q2[:, 1]) / sg
    # s¹-动量
    R2 = (g_q1[:, 2] + g_F11[:, 0] + g_F12[:, 1]) / sg + Ch1 + S1
    # s²-动量
    R3 = (g_q2[:, 2] + g_F12[:, 0] + g_F22[:, 1]) / sg + Ch2 + S2

    return R1, R2, R3, eta, geo


# ================================================================
# 5. 损失函数（严格三部分，与 combined2 相同）
# ================================================================
def total_loss(net, Xp, Xi, Xb):
    """
    L = L_pde + W_fs · L_fs + W_ibc · (L_i + L_b)
      L_pde = mean(R1² + R2² + R3²)
      L_fs  = mean((η N + B − ζ0)²)          （良平衡全域约束, Eq.(2.21)）
      L_i   = mean((η−η0)² + q1² + q2²) @t=0 （纯静水初值）
      L_b   = mean(q1² + q2²) @∂Γ            （固壁: 法向流量为零）
    """
    # ---- 第一部分: 控制方程误差 ----
    R1, R2, R3, eta_p, geo_p = residuals(Xp, net)
    L_pde = torch.mean(R1**2 + R2**2 + R3**2)

    # ---- 第二部分: 自由水面误差 ----
    Bp, Np = geo_p[0], geo_p[4]
    L_fs = torch.mean((eta_p * Np + Bp - ZETA0)**2)

    # ---- 第三部分: 初边值条件误差 ----
    # 初始条件目标: 纯静水态 η = η0 = (ζ0-B)/N, q1 = q2 = 0
    eta_i, q1i, q2i = net(Xi)
    xi, yi = Xi[:, 0], Xi[:, 1]
    geo_i = geometry_monge(xi, yi)
    B_i, N_i = geo_i[0], geo_i[4]
    eta_init = (ZETA0 - B_i) / N_i          # 静水法向水深 (目标值)
    L_i = torch.mean((eta_i - eta_init)**2 + q1i**2 + q2i**2)

    # 边界条件: 固壁, 法向流量为零（按 combined2: 直接约束 q 分量）
    _, q1b, q2b = net(Xb)
    L_b = torch.mean(q1b**2 + q2b**2)

    L_ibc = L_i + L_b
    L_total = L_pde + W_FS * L_fs + W_IBC * L_ibc

    return L_total, L_pde, L_fs, L_ibc


# ================================================================
# 6. 配点生成（与 combined2 的 make_points 逐点相同: 张量积网格 + 四边）
# ================================================================
def make_points(nx=None, ny=None, nt=None, m=None):
    """
    内部配点 Xp: nx × ny × nt (时空张量积)
    初始点 Xi:   nx × ny (t=0)
    边界点 Xb:   四条边各 m 点 × nt 时间层
    缺省(None)时取当前全局 NX/NY/NT/M_BC —— 在调用时解析（非 def 时绑定），
    以便 --fast 修改全局后真正生效（修默认参数绑定 bug）。
    """
    nx = NX if nx is None else nx
    ny = NY if ny is None else ny
    nt = NT if nt is None else nt
    m  = M_BC if m is None else m
    xv = torch.linspace(0, LX, nx, dtype=torch.float64)
    yv = torch.linspace(0, LY, ny, dtype=torch.float64)
    tv = torch.linspace(0, T_END, nt, dtype=torch.float64)

    Xg, Yg, Tg = torch.meshgrid(xv, yv, tv, indexing="ij")
    Xp = torch.stack([Xg, Yg, Tg], dim=-1).reshape(-1, 3)

    # 初始点 (t=0)
    Xi = torch.stack([Xg[..., 0], Yg[..., 0],
                      torch.zeros_like(Tg[..., 0])], dim=-1).reshape(-1, 3)

    # 边界点 (四条边)
    e  = torch.linspace(0, LX, m, dtype=torch.float64)
    z0 = torch.zeros(m, dtype=torch.float64)
    xyb = torch.cat([
        torch.stack([e, z0], dim=-1),
        torch.stack([e, torch.full((m,), LY, dtype=torch.float64)], dim=-1),
        torch.stack([z0, e], dim=-1),
        torch.stack([torch.full((m,), LX, dtype=torch.float64), e], dim=-1)
    ], dim=0)
    tb = torch.linspace(0, T_END, nt, dtype=torch.float64)
    Xb = torch.cat([
        xyb.repeat_interleave(nt, dim=0),
        tb.repeat(xyb.shape[0]).unsqueeze(-1)
    ], dim=1)

    return Xp, Xi, Xb


# ================================================================
# 7. 良平衡审计（论文 Eq.(2.21) + Prop.2.5；附加输出，不参与训练）
# ================================================================
def audit_well_balanced(Xp, tag=""):
    """
    Monge 度量静水态 η=(ζ0−B)/N = (ζ0−B)/γ, q=0 下:
      (i)  完整装配残差 R2, R3 ≈ 0
      (ii) 恒等式 R^j = g·η·g^{ji}∂_i ζ ≈ 0（Prop.2.5 良平衡的强形式）
      (iii) ζ = ηN + B ≡ ζ0（Eq.(2.21)）
    """
    X = Xp.clone().requires_grad_(True)
    x, y = X[:, 0], X[:, 1]
    geo = geometry_monge(x, y)
    (B, Bx, By, sg, N, N1, N2, g11, g22, g12,
     G1_11, G1_12, G1_22, G2_11, G2_12, G2_22) = geo
    eta = (ZETA0 - B) / N
    q1 = torch.zeros_like(eta)
    q2 = torch.zeros_like(eta)

    F11, F12, F22, Ch1, Ch2, S1, S2 = flux_and_source(eta, q1, q2, geo)

    def gd(v):
        return torch.autograd.grad(v.sum(), X, create_graph=True)[0]

    g_F11 = gd(sg * F11); g_F12 = gd(sg * F12); g_F22 = gd(sg * F22)
    R2 = (g_F11[:, 0] + g_F12[:, 1]) / sg + Ch1 + S1
    R3 = (g_F12[:, 0] + g_F22[:, 1]) / sg + Ch2 + S2

    zeta = eta * N + B
    dzdx = torch.autograd.grad(zeta.sum(), x, create_graph=True)[0]
    dzdy = torch.autograd.grad(zeta.sum(), y, create_graph=True)[0]
    R2_h = G_GRAV * eta * (g11 * dzdx + g12 * dzdy)   # Monge: 含 g12 交叉项
    R3_h = G_GRAV * eta * (g12 * dzdx + g22 * dzdy)

    m = lambda v: float(v.detach().abs().max())
    print(f"  [审计:{tag}] 静水态残差: max|R2| = {m(R2):.3e}, "
          f"max|R3| = {m(R3):.3e}")
    print(f"  [审计:{tag}] 恒等式: max|R2−R2_hydro| = {m(R2 - R2_h):.3e}, "
          f"max|R3−R3_hydro| = {m(R3 - R3_h):.3e}, "
          f"max|ζ−ζ0| = {m(zeta - ZETA0):.3e}")
    ok = (m(R2) < 1e-10 and m(R3) < 1e-10
          and m(R2 - R2_h) < 1e-10 and m(R3 - R3_h) < 1e-10)
    print(f"  [审计:{tag}] Monge 版良平衡{'通过 ✅' if ok else '失败 ❌'}")
    return ok


# ================================================================
# 8. 训练 + 评估 + 可视化（与 combined2 的 main() 流程一致）
# ================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="ISWE-PINN（Monge 度量版；结构/参数与 pinn_swe_combined2.py 相同）")
    p.add_argument("--fast", action="store_true",
                   help="小配点快速冒烟测试（仅测试用；默认与 combined2 参数一致）")
    p.add_argument("--no-plot", action="store_true",
                   help="不生成图片（仅数值输出）")
    p.add_argument("--outdir", type=str, default=None,
                   help="输出目录（默认 = 脚本所在目录，即本代码文件夹）")
    p.add_argument("--device", type=str, default="auto",
                   help="计算设备: auto（默认，有 CUDA 则用 GPU）/ cpu / cuda / cuda:N")
    return p.parse_args()


def main():
    args = parse_args()

    global N_ITER, NX, NY, NT, M_BC, WIDTH, LAYERS, LBFGS_MAXITER, LBFGS_MAXFUN, DEVICE
    DEVICE = resolve_device(args.device)
    if args.fast:
        # --fast 仅用于快速冒烟：缩小配点/网络/迭代，使分钟级跑完。
        # 默认（无 --fast）严格保持 combined2 值不变。
        N_ITER = 100
        NX, NY, NT, M_BC = 12, 12, 5, 12
        WIDTH, LAYERS = 40, 3
        LBFGS_MAXITER, LBFGS_MAXFUN = 30, 200

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # 默认输出到脚本所在目录（即本代码文件夹），便于与代码同放
    outdir = args.outdir or os.path.dirname(os.path.abspath(__file__))
    os.makedirs(outdir, exist_ok=True)

    # 设备：网络与全部配点张量搬到 DEVICE（strict 审核 P0 修复）
    net = SWENet(width=WIDTH, layers=LAYERS).to(DEVICE)
    Xp, Xi, Xb = make_points()
    Xp, Xi, Xb = Xp.to(DEVICE), Xi.to(DEVICE), Xb.to(DEVICE)

    # ---- 良平衡审计（训练前）----
    audit_well_balanced(Xp, tag="训练前")

    # 优化器: Adam + 指数衰减（与 combined2 相同: 1e-3, decay_steps=800, rate=0.8）
    gamma = 0.8 ** (1.0 / 800.0)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    # ---- 误差历史记录 ----
    hist_it, hist_total, hist_pde, hist_fs, hist_ibc = [], [], [], [], []

    # ---- 逐次误差日志（每次迭代写一行；文件落在 outdir 即代码文件夹）----
    err_path = os.path.join(outdir, "iswe_pinn_monge_errors.txt")
    err_fh = open(err_path, "w", encoding="utf-8")
    err_fh.write("# ISWE-PINN Monge 度量版 —— 逐次迭代误差日志\n")
    err_fh.write("# 阶段 iter  total  pde  fs  ibc\n")
    err_fh.flush()
    # 异常退出时也尽量落盘（严格审核 P2#11）
    import atexit
    atexit.register(lambda: err_fh.close() if not err_fh.closed else None)

    print("=" * 72)
    print("  ISWE-PINN Monge 度量版 —— 训练开始")
    print("=" * 72)
    print(f"  内部配点: {Xp.shape[0]:5d}  初始点: {Xi.shape[0]:4d}  "
          f"边界点: {Xb.shape[0]:4d}")
    print(f"  迭代次数: {N_ITER}  时间域: [0, {T_END}]s  "
          f"域: [0,{LX}]×[0,{LY}]")
    print(f"  损失: L_pde + {W_FS}·L_fs + {W_IBC}·(L_i+L_b)")
    print(f"  初始条件: 纯静水 η0=(ζ0-B)/N, q=0  "
          f"(扰动参数化 δ0={INIT_BIAS}, 起点{'精确静水态' if INIT_BIAS == 0 else '非精确'})")
    print("  方程构造: Monge 度量（Eq.G/Ginv/ChrisClosed/divT_comp/U/S），g^{12}≠0")
    print(f"  计算设备: {DEVICE}"
          + (f"  ({torch.cuda.get_device_name(DEVICE)})" if DEVICE.type == "cuda" else ""))
    print("=" * 72)

    for it in range(N_ITER):
        optimizer.zero_grad()
        L_total, L_pde, L_fs, L_ibc = total_loss(net, Xp, Xi, Xb)
        L_total.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)   # clip_by_global_norm=1.0
        optimizer.step()
        sched.step()

        # 逐次写盘（Adam 阶段: 每次迭代）
        err_fh.write("adam %d %.12e %.12e %.12e %.12e\n"
                     % (it, float(L_total.detach()), float(L_pde.detach()),
                        float(L_fs.detach()), float(L_ibc.detach())))

        if it % RECORD_EVERY == 0:
            hist_it.append(it)
            hist_total.append(float(L_total.detach()))
            hist_pde.append(float(L_pde.detach()))
            hist_fs.append(float(L_fs.detach()))
            hist_ibc.append(float(L_ibc.detach()))

        if it % LOG_EVERY == 0:
            print(f"  iter {it:5d} | total {float(L_total.detach()):.4e} | "
                  f"pde {float(L_pde.detach()):.4e} | fs {float(L_fs.detach()):.4e} | "
                  f"ibc {float(L_ibc.detach()):.4e}")

    # ================================================================
    # 第二阶段: scipy L-BFGS-B 精细收敛
    #   maxiter=30000, maxfun=100000, ftol=gtol=1e-12, strong-Wolfe 线搜索
    # ================================================================
    print("\n" + "=" * 72)
    print("  切换 L-BFGS 精细收敛 (strong-Wolfe 线搜索, 上限 30000 iterations)")
    print("=" * 72)

    from scipy.optimize import minimize as scipy_minimize

    params = list(net.parameters())
    shapes  = [p_.shape for p_ in params]
    sizes   = [int(np.prod(s)) for s in shapes]
    offsets = np.cumsum([0] + sizes)

    def flatten_params(ps):
        return (torch.cat([p.reshape(-1) for p in ps])
                .detach().cpu().numpy().astype(np.float64))

    def unflatten_assign(flat):
        # scipy 传进来的是 numpy（CPU）；建到 DEVICE 上再 copy_
        ft = torch.tensor(flat, dtype=torch.float64, device=DEVICE)
        with torch.no_grad():
            for i, p_ in enumerate(params):
                p_.copy_(ft[offsets[i]:offsets[i + 1]].reshape(shapes[i]))

    def value_and_grad(flat_np):
        unflatten_assign(flat_np)
        for p_ in params:
            if p_.grad is not None:
                p_.grad = None
        L_total, _, _, _ = total_loss(net, Xp, Xi, Xb)
        L_total.backward()
        # 按变量顺序拼梯度（grad 为 None 的变量补零，防 flatten 崩溃）
        g_flat = (torch.cat([(p_.grad if p_.grad is not None
                              else torch.zeros_like(p_)).reshape(-1)
                             for p_ in params])
                  .detach().cpu().numpy().astype(np.float64))
        return float(L_total.detach()), g_flat

    # 损失值（float）评估：内部对结果 .detach()，可在无 no_grad 下安全调用
    def total_loss_val(net_, Xp_, Xi_, Xb_):
        L_total, L_pde, L_fs, L_ibc = total_loss(net_, Xp_, Xi_, Xb_)
        return (float(L_total.detach()), float(L_pde.detach()),
                float(L_fs.detach()), float(L_ibc.detach()))

    lbfgs_count = {"n": 0}

    def lbfgs_callback(xk):
        lbfgs_count["n"] += 1
        unflatten_assign(xk)
        # 注意：不可用 torch.no_grad() 包裹 —— residuals 内部依赖
        # autograd.grad(create_graph=True) 求 PDE 残差，禁用梯度会 RuntimeError。
        # total_loss_val 已对返回值 .detach()，此处安全（与 combined2 callback 一致）。
        L_total, L_pde, L_fs, L_ibc = total_loss_val(net, Xp, Xi, Xb)
        it_lb = N_ITER + lbfgs_count["n"]
        # 逐次写盘（L-BFGS 阶段: 每次函数评估/迭代）
        err_fh.write("lbfgs %d %.12e %.12e %.12e %.12e\n"
                     % (it_lb, L_total, L_pde, L_fs, L_ibc))
        if lbfgs_count["n"] % LOG_EVERY == 0:
            print(f"  iter {it_lb:5d} | total {L_total:.4e} | "
                  f"pde {L_pde:.4e} | fs {L_fs:.4e} | ibc {L_ibc:.4e}")
        if lbfgs_count["n"] % RECORD_EVERY == 0:
            hist_it.append(it_lb)
            hist_total.append(L_total)
            hist_pde.append(L_pde)
            hist_fs.append(L_fs)
            hist_ibc.append(L_ibc)

    x0 = flatten_params(params)
    lbfgs_result = scipy_minimize(
        value_and_grad, x0, method="L-BFGS-B", jac=True,
        callback=lbfgs_callback,
        options={"maxiter": LBFGS_MAXITER, "maxfun": LBFGS_MAXFUN,
                 "ftol": 1e-12, "gtol": 1e-12})

    unflatten_assign(lbfgs_result.x)

    L_lb, Lp_lb, Lfs_lb, Libc_lb = total_loss_val(net, Xp, Xi, Xb)
    final_it = N_ITER + int(lbfgs_result.nit)
    if hist_it[-1] < final_it:
        hist_it.append(final_it)
        hist_total.append(L_lb)
        hist_pde.append(Lp_lb)
        hist_fs.append(Lfs_lb)
        hist_ibc.append(Libc_lb)

    print(f"  L-BFGS 完成: success={bool(lbfgs_result.success)}, "
          f"iterations={int(lbfgs_result.nit)}, "
          f"objective={float(lbfgs_result.fun):.4e}")
    print(f"  精细收敛后损失: total={L_lb:.4e}  pde={Lp_lb:.4e}  "
          f"fs={Lfs_lb:.4e}  ibc={Libc_lb:.4e}")

    # ---- 间隔误差表（按 RECORD_EVERY 记录；同旧 error_vs_iteration.txt 口径）----
    err_fh.close()
    with open(os.path.join(outdir, "error_vs_iteration.txt"), "w",
              encoding="utf-8") as fh:
        fh.write("# iteration total_loss pde_loss fs_loss ibc_loss\n")
        for _it, _t, _p, _f, _i in zip(hist_it, hist_total, hist_pde,
                                       hist_fs, hist_ibc):
            fh.write("%d %.12e %.12e %.12e %.12e\n" % (_it, _t, _p, _f, _i))
    print(f"  误差记录已写: {err_path}")
    print(f"  间隔误差表已写: {os.path.join(outdir, 'error_vs_iteration.txt')}")

    # ---- 良平衡审计（训练后，对解析静水态；训练态见下方 ζ 误差）----
    audit_well_balanced(Xp, tag="训练后")

    # ================================================================
    # 评估 (t = T_END 精细网格, ne=120 —— 与 combined2 相同)
    # ================================================================
    print("\n" + "=" * 72)
    print("  训练完成 —— 评估与可视化")
    print("=" * 72)

    ne = 120
    xe = torch.linspace(0, LX, ne, dtype=torch.float64)
    XE, YE = torch.meshgrid(xe, xe, indexing="ij")
    Xev = torch.stack([XE.reshape(-1), YE.reshape(-1),
                       torch.full((XE.numel(),), T_END, dtype=torch.float64)],
                      dim=-1).to(DEVICE)

    with torch.no_grad():
        geo_ev = geometry_monge(Xev[:, 0], Xev[:, 1])
        B_ev, N_ev = geo_ev[0], geo_ev[4]
        eta_ev, q1_ev, q2_ev = net(Xev)

        zeta_np = (eta_ev * N_ev + B_ev).cpu().numpy().reshape(ne, ne)
        eta_np  = eta_ev.cpu().numpy().reshape(ne, ne)
        B_np    = B_ev.cpu().numpy().reshape(ne, ne)

        h_ev = torch.clamp(eta_ev, min=1e-8)
        u1_ev = (q1_ev / h_ev).cpu().numpy().reshape(ne, ne)
        u2_ev = (q2_ev / h_ev).cpu().numpy().reshape(ne, ne)
        u_max = float(np.sqrt(u1_ev**2 + u2_ev**2).max())

    Lf, Lpf, Lfsf, Libcf = total_loss_val(net, Xp, Xi, Xb)

    print(f"  t = {T_END}s 评估结果:")
    print(f"    max|ηN+B - {ZETA0}| = {np.abs(zeta_np - ZETA0).max():.4e}")
    print(f"    max|u|               = {u_max:.4e}")
    print(f"    η  range: [{eta_np.min():.4f}, {eta_np.max():.4f}]")
    print(f"    ζ  range: [{zeta_np.min():.4f}, {zeta_np.max():.4f}]")
    print(f"  最终损失: total={Lf:.4e}  pde={Lpf:.4e}  "
          f"fs={Lfsf:.4e}  ibc={Libcf:.4e}")

    np.savez(os.path.join(outdir, "iswe_pinn_result_monge.npz"),
             zeta=zeta_np, eta=eta_np, B=B_np, u1=u1_ev, u2=u2_ev,
             hist_it=np.array(hist_it), hist_total=np.array(hist_total),
             hist_pde=np.array(hist_pde), hist_fs=np.array(hist_fs),
             hist_ibc=np.array(hist_ibc))

    if args.no_plot:
        print("=" * 72)
        print("  全部完成（--no-plot：未生成图片）。")
        print("=" * 72)
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm

    XE_np, YE_np = XE.numpy(), YE.numpy()

    # ---- 图1: t=T_END 预测自由水面 η cos<n,z> + B ----
    fig1 = plt.figure(figsize=(10, 7))
    ax1 = fig1.add_subplot(111, projection="3d")
    surf1 = ax1.plot_surface(XE_np, YE_np, zeta_np, cmap=cm.viridis,
                             linewidth=0, antialiased=True)
    ax1.set_title(
        rf"Predicted Free Surface $\eta\cos\langle n,z\rangle+B$ at $t={T_END}$s",
        fontsize=13)
    ax1.set_xlabel("$x$", fontsize=11)
    ax1.set_ylabel("$y$", fontsize=11)
    ax1.set_zlabel(r"$\zeta$", fontsize=11)
    fig1.colorbar(surf1, ax=ax1, shrink=0.6, pad=0.1)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "pinn_free_surface_monge.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [图1] pinn_free_surface_monge.png  —— t={T_END}s 预测自由水面")

    # ---- 图2: 误差随迭代变化 (四曲线) ----
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.semilogy(hist_it, hist_total, "-",  lw=1.8, color="black",
                 label="Total Loss $L$")
    ax2.semilogy(hist_it, hist_pde,   "--", lw=1.3, color="#1f77b4",
                 label=r"PDE Residual $L_{\mathrm{pde}}$")
    ax2.semilogy(hist_it, hist_fs,    "-.", lw=1.3, color="#d62728",
                 label=r"Free Surface Error $L_{\mathrm{fs}}$")
    ax2.semilogy(hist_it, hist_ibc,   ":",  lw=1.3, color="#2ca02c",
                 label=r"IC/BC Error $L_{\mathrm{ibc}}$")
    ax2.set_xlabel("Iteration", fontsize=12)
    ax2.set_ylabel("Error (log scale)", fontsize=12)
    ax2.set_title("Convergence: Error vs. Iteration", fontsize=13)
    ax2.grid(True, which="both", ls=":", alpha=0.6)
    ax2.legend(fontsize=11, loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "pinn_convergence_monge.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  [图2] pinn_convergence_monge.png   —— 误差随迭代变化")

    # ---- 图3: 地形图 B(x,y) ----
    fig3 = plt.figure(figsize=(10, 7))
    ax3 = fig3.add_subplot(111, projection="3d")
    surf3 = ax3.plot_surface(XE_np, YE_np, B_np, cmap=cm.terrain,
                             linewidth=0, antialiased=True)
    ax3.set_title(
        r"Bottom Topography $B=0.3\sin(\pi x)\cos(\pi y)$",
        fontsize=13)
    ax3.set_xlabel("$x$", fontsize=11)
    ax3.set_ylabel("$y$", fontsize=11)
    ax3.set_zlabel("$B$", fontsize=11)
    fig3.colorbar(surf3, ax=ax3, shrink=0.6, pad=0.1)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "pinn_topography_monge.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  [图3] pinn_topography_monge.png    —— 地形图")

    if DEVICE.type == "cuda":
        print(f"  [GPU] 峰值显存: {torch.cuda.max_memory_allocated(DEVICE)/2**20:.1f} MiB"
              f"  (device={torch.cuda.get_device_name(DEVICE)})")
    print("=" * 72)
    print("  全部完成。")
    print("=" * 72)


if __name__ == "__main__":
    main()
