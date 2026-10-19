#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
E2 / Eq.(27): 圆域溃坝（间断解）的内蕴浅水方程 PINN，N-LAAF 激活
================================================================================
论文公式对照（manuscript.tex）
  Eq.(2)            正交 LCS 标架 t_1, t_2, t_3（Gram-Schmidt）
  Eq.(3)            诱导度量 G_sw = diag(h_(1)^2, h_(2)^2)；逆度量 g^{ij} = (G_sw^{-1})^{ij}
  Eq.(4)            J = sqrt(det G_sw) = h_(1) h_(2)
  Eq.(5)            gamma = d x^3 / d s^3 = 1/J = 1/sqrt(1+|grad z|^2)
  Eq.(7)            度量感知算子
                      (grad_G f)^i      = g^{ij} d_j f
                      (div_G v)         = (1/J) d_i(J v^i)
                      (div_G T)^j       = (1/J) d_i(J T^{ij}) + Gam^j_ik T^{ik}
  Eq.(9)            内蕴浅水方程（保守形式）
  Eq.(12)           网络输出 (h, u^1, u^2)：直接以**法向水深 h** 为未知量（溃坝无静水表面，故不用 Eq.(13) 拟设）
  Eq.(14)(15)       质量/动量残差（hybrid 原始--保守装配）
  Eq.(16)           复合损失
  Eq.(17)(18)(19)   三张床 p1 / p2 / p3（另加零床 flat 做退化检验）
  Eq.(27)           圆域溃坝初始条件 h = 2 (圆盘内), 1 (其它), u = v = 0, 域 [-10,10]^2
  Jagtap2020        N-LAAF 激活 + slope recovery 项

与 exp_bedp1/iswe_pinn_e1_bedp1_lake.py 的关系（重要，也是本脚本可信度的来源）
  (a) geometry()    —— Eq.(2)(3)(4)(5) 的 h_(i), J, gamma, d_j gamma, g^{ij},
                        六个 Christoffel；公式与 geometry_bedp1() **逐行相同**，
                        只把床 jet 换成可切换的 BED_JET[BED]。
  (b) flux_and_source() 与 residuals() 的装配行 —— 与成功脚本 **逐字相同**。
                        唯一差别：residuals() 额外返回 u1, u2（供干湿罚使用），
                        装配行一字未改。
  (c) 该装配已通过三重审计：sympy 符号级 41/41、源码级保真 62/62（ast 直读源码）、
      numpy 独立有限差分 21/21，以及服务器实测（静水态 max|R2| = 3.553e-15,
      max|R3| = 0）。
  本脚本相对成功脚本只改三处：
      (1) 网络：固定 tanh  ->  N-LAAF（逐神经元可训练斜率 + slope recovery 项）
      (2) 初值：静水态     ->  间断台阶 Eq.(27)，初值罚按 Eq.(16) 的 ||U - U^ic||^2
      (3) 边界：速度为零   ->  固壁 q.n = 0（默认）或 Neumann（圆域溃坝无解析解）
  新增一个**精确**诊断量：总质量 M = ∫_U h J dx dy。由 Eq.(14) + 固壁条件，
      dM/dt = -∮ J q.n ds = 0  =>  M 精确守恒，是无需参考解的独立判据。

slope recovery（Jagtap2020；已核对 arXiv:1909.12228v4 第 4 页原文）
      S(a) = 1 / [ (1/(D-1)) * sum_{k=1}^{D-1} exp( (1/N_k) * sum_i a_i^k ) ]
  注意是**倒数**。最小化 w_a*S(a) 会把激活斜率推**大**（原文动机：
  "increase the slope of activation function, resulting in non-vanishing gradients"）。
  旧稿 <home>/间断解/ddb8_pinn.py 的 slope_recovery() 只算了分母（漏了倒数），
  方向恰好相反；详见 AUDIT.md 第 4 节。

用法
  python iswe_pinn_e2_circulardam.py --audit-only      # 只跑审计不训练（数秒）
  python iswe_pinn_e2_circulardam.py --fast            # 小配点冒烟
  python iswe_pinn_e2_circulardam.py                   # 默认：Eq.(27) 原样（零床）
  python iswe_pinn_e2_circulardam.py --bed p3 --r0 1.2 # 全三维床上
"""
import argparse
import os
import time
from collections import namedtuple

import numpy as np
import torch

torch.set_default_dtype(torch.float64)

# ---- 计算设备  # <<< DEVICE v1 >>>
# 由 --device 覆盖。**仅当设备为 cuda 时**才启用 torch.set_default_device，
# 故 cpu 模式（默认）下张量创建与数值路径与既往版本逐字相同（零回归）。
DEVICE = torch.device("cpu")

# ============================================================================
# 0. 全局物理与计算参数（全部可由命令行覆盖，见 apply_cli）
# ============================================================================
G_GRAV = 9.81           # 重力加速度 g                       [Eq.(9)]

BED = "flat"            # 床的选择，见 BED_JET / BED_DEFAULTS
X0, X1 = -10.0, 10.0    # 域 U = [X0,X1] x [Y0,Y1]
Y0, Y1 = -10.0, 10.0
XC, YC = 0.0, 0.0        # 溃坝圆心（默认取域中心）
R0 = 3.0                 # 溃坝半径；Eq.(27) 为 3，使 x^2 + y^2 <= 9 逐字成立
H_IC, L_IC = 2.0, 1.0    # 台阶内 / 外的初始水深            [Eq.(27)]
T_END = 1.0              # 时间区间 [0, T]，与 Tian2025 的 dDB8 一致

# <<< SHOCK_SNAP v2 >>>
# ---- 激波传播快照时刻：**整个 [0, T] 上每 0.1 s 一帧** ----
#   T_SNAP = None 时自动取 0.1 的整数倍直到 T_END（T_END = 1.0 -> 10 帧：
#   0.1, 0.2, ..., 1.0）。每帧出三张图（平面水深场 / 3D 水面高度 / 径向剖面），
#   全部落在服务器 outdir 里，便于远程直接查看波的传播过程。
T_SNAP = None


# ---- 训练参数（默认对齐 Tian2025 的 dDB8：间断初值需要大权重）----
N_ITER = 3000            # Adam 迭代步数
LOG_EVERY = 50
RECORD_EVERY = 50
W_F = 1.0                # omega_f  残差项                   [Eq.(16)]
W_I = 100.0              # omega_I  初值项
W_B = 1.0                # omega_b  边界项                   [Eq.(16)]
W_POS = 10.0             # L_pos 非负性 + 干湿罚              [Eq.(16)]
W_A = 0.0                # omega_a  slope recovery 权重。        # <<< REFORM v1 >>>
                         # 置 0 = 从损失里**移除** S(a)，对齐 Tian2025 的仓库代码
                         # （src/model/laaf.py 只有可训练斜率 a，没有 slope-recovery
                         # 损失项）。原 W_A = 1 时 S(a) 在 E2 终态**独占损失的
                         # 99.99%**（L = 3.0361e-01，其中 S(a) = 0.3036），优化预算
                         # 被一项与物理无关的正则吃掉。N-LAAF 本体不变：斜率 a 仍
                         # 是可训练参数、仍参与激活，只是不再计入损失。可用 --wa 覆盖。

N_SCALE = 10.0           # N-LAAF 缩放因子 n（Jagtap2020 函数逼近算例取 10）
H_DW = 1.0e-2            # 干湿罚的水深阈值。原 h 拟设下该项恒为 0；删去 H_0 后
                         # h 由网络直接输出、初始可略为负，故权重在 h<0 处封顶为 1
                         # （见 total_loss 注释）。
LBFGS_MAXITER = 30000
LBFGS_MAXFUN = 100000
SEED = 42

NX, NY, NT, M_BC = 60, 60, 10, 60    # 仅 sampler == "grid"（旧张量网格）时使用
# <<< REFORM v1 >>> 配点采样：消除**时域混叠**
#   旧版内部配点是 60x60x10 的**张量网格**，t 方向只有 10 个离散平面，
#   任意两个相邻平面之间 0 个点。网络可以在空档里塞进一段宽 delta << dt 的
#   陡过渡：配点上 d_t h ~ 0（残差被"满足"），空档里 d_t h ~ h/delta 极大。
#   E2-flat 的 M(t) 在 t = 0.25 / 0.5 / 0.75 s 崩 6% / 50% / 99%，而
#   L_pde = 1.9e-6、L_bc = 2.0e-6 **全部"满足"** —— 正是这一机理
#   （见审核报告 AUDIT_E2_vs_dCDB9_20261003.md 第 3 节与构造反例）。
#   改为 3D 低差异（Sobol, scramble）采样后 t 方向边际非退化，混叠被消除。
#   配点在 **CPU** 上用 numpy/scipy 生成再搬到 DEVICE，故 --device cpu 与
#   --device cuda 的配点逐位相同（CUDA RNG 与 CPU RNG 实现不同）。
N_INT = 32768            # 内部配点数（Sobol 3D，域 U x [0,T]；2^15，
                         # 使 Sobol 的平衡性质成立；并与 Tian dCDB9 的
                         # 32768 个内部配点逐字对齐。禁用 65536：float64 全批
                         # 反传峰值 15 GiB/进程，与本项目另一训练任务同卡即 OOM）
N_IC = 4096              # 初始面 t = 0 的配点数（Sobol 2D）
N_BC_SIDE = 2048         # 每个边界面的配点数（4 个面 -> Nb = 8192）
SAMPLER = "sobol"        # sobol（默认）/ grid（旧张量网格，A/B 对照见 --sampler）
W_PDE_M = 10.0           # L_pde 中质量方程 R1 的权重（对齐 Tian dCDB9 的 10:1）
WIDTH, LAYERS = 128, 6
BC_MODE = "wall"         # wall | zgrad | periodic

REBALANCE = False        # Wang2021 式梯度重平衡（默认关，见 grad_norm_monitor）
RB_EVERY = 500           # 监控/重平衡的间隔步

# 几何量容器（与成功脚本的 17 元组同序，便于逐项核对）
Geo = namedtuple("Geo", ["B", "Bx", "By", "J", "N", "N1", "N2",
                         "g11", "g22", "g12",
                         "G1_11", "G1_12", "G1_22",
                         "G2_11", "G2_12", "G2_22"])


# ============================================================================
# 1. 床的 jet：z 与 z_1, z_2, z_11, z_12, z_22 的闭式（全部精确、无差分）
# ============================================================================
def _zero_like(x):
    """值恒为 0，但**保持在 autograd 计算图内**。

    torch.zeros_like(x) 与输入 X 没有计算图连接，对纯床量（geo.B、h_(1)、
    ln J）求 ∂ 会抛 "element 0 of tensors does not require grad"。审计要用
    autograd 求 ∂_i z 等，故零量写成 x * 0.0 —— 值完全不变，但在图里。
    """
    return x * 0.0


def bed_flat(x, y):
    """z = 0：gamma = 1, G_sw = I, 所有 Christoffel = 0（退化检验用）。"""
    z = _zero_like(x)
    return z, z, z, z, z, z


def bed_p1(x, y):
    """Eq.(17): z = (1/25)(x-10)^2（论文声明域 [0,10]x[0,1]）。"""
    zero = _zero_like(x)
    return ((x - 10.0) ** 2 / 25.0,        # z
            2.0 * (x - 10.0) / 25.0,       # z_1
            zero,                          # z_2
            torch.full_like(x, 2.0 / 25.0),  # z_11
            zero,                          # z_12
            zero)                          # z_22


def bed_p2(x, y):
    """Eq.(18): z = -(4/5) sqrt(x^2 + y^2 + 1)（论文声明域 [-3,3]^2）。

    s = sqrt(r2), r2 = x^2 + y^2 + 1:
      z_1  = -(4/5) x/s                 z_2  = -(4/5) y/s
      z_11 = -(4/5)(s^2 - x^2)/s^3      z_22 = -(4/5)(s^2 - y^2)/s^3
      z_12 =  (4/5) x y / s^3
    """
    r2 = x ** 2 + y ** 2 + 1.0
    s = torch.sqrt(r2)
    c = 0.8
    return (-c * s,
            -c * x / s,
            -c * y / s,
            -c * (s ** 2 - x ** 2) / s ** 3,
            c * x * y / s ** 3,
            -c * (s ** 2 - y ** 2) / s ** 3)


def bed_p3(x, y):
    """Eq.(19): z = -(1/500) x^3 - (1/100) x y^2（论文声明域 [-10,10]x[-4,4]）。"""
    return (-(x ** 3) / 500.0 - x * (y ** 2) / 100.0,   # z
            -3.0 * x ** 2 / 500.0 - (y ** 2) / 100.0,   # z_1
            -x * y / 50.0,                              # z_2
            -6.0 * x / 500.0,                           # z_11
            -y / 50.0,                                  # z_12
            -x / 50.0)                                  # z_22


BED_JET = {"flat": bed_flat, "p1": bed_p1, "p2": bed_p2, "p3": bed_p3}

# 各床在论文中的声明域，以及推荐的溃坝半径
# r0 = 0.3 * min(半宽, 半高)，使圆心到最近边界距离 = 3.33 r0；
# flat 用 Eq.(27) 的原值 3.0，使 x^2+y^2 <= 9 逐字成立。
BED_DEFAULTS = {
    "flat": dict(domain=(-10.0, 10.0, -10.0, 10.0), r0=3.0,
                 note="Eq.(27) 原始设定（零床，退化为经典 SWE）"),
    "p1":   dict(domain=(0.0, 10.0, 0.0, 1.0), r0=0.15,
                 note="细长条域 10:1，圆域溃坝在此不适宜；仅为完整性保留"),
    "p2":   dict(domain=(-3.0, 3.0, -3.0, 3.0), r0=0.9,
                 note="HCB 中心凸床"),
    "p3":   dict(domain=(-10.0, 10.0, -4.0, 4.0), r0=1.2,
                 note="全三维床"),
}


# ============================================================================
# 2. 解析几何量：正交 LCS（论文 Eq.(2)(3)(4)(5)，尺度因子显式 Christoffel）
#    *** 公式与 iswe_pinn_e1_bedp1_lake.py::geometry_bedp1 逐行相同 ***
#    *** 唯一区别：床 jet 来自 BED_JET[BED]，不再写死为床 (17)      ***
# ============================================================================
def geometry(x, y):
    r"""返回 Geo(...)；其中 N 即论文的 gamma，J = sqrt(det G_sw) = 1/N。"""
    B, Bx, By, Bxx, Bxy, Byy = BED_JET[BED](x, y)

    # ---- 尺度因子 Eq.(2)/(3a-c)：Gram-Schmidt 正交标架的范数 ----
    #   t_1 = [1, 0, z_1]                     => h_(1)^2 = 1 + z_1^2
    #   t_2 = [-z_1 z_2/(1+z_1^2), 1, z_2/(1+z_1^2)]
    #                                         => h_(2)^2 = (1+|grad z|^2)/(1+z_1^2)
    h1_2 = 1.0 + Bx ** 2
    h2_2 = (1.0 + Bx ** 2 + By ** 2) / h1_2
    h1 = torch.sqrt(h1_2)
    h2 = torch.sqrt(h2_2)

    # ---- J 与 gamma  Eq.(4)(5) ----
    J = h1 * h2
    N = 1.0 / J

    # ---- d_j gamma = -gamma^3 (z_i z_ij) ----
    N1 = -(N ** 3) * (Bx * Bxx + By * Bxy)         # d_1 gamma
    N2 = -(N ** 3) * (Bx * Bxy + By * Byy)         # d_2 gamma

    # ---- 逆度量 Eq.(3)：G_sw^{-1} = diag(h_(1)^{-2}, h_(2)^{-2}) ----
    g11 = 1.0 / h1_2
    g22 = 1.0 / h2_2
    g12 = torch.zeros_like(g11)                    # 正交 => g^{12} ≡ 0（精确，非近似）

    # ---- Christoffel（尺度因子显式展开；恒等于 diag(h_(1)^2, h_(2)^2) 的标准式）----
    dh1_dx = Bx * Bxx / h1                         # d_1 h_(1)
    dh1_dy = Bx * Bxy / h1                         # d_2 h_(1)
    dh2_2_dx = (2.0 * (Bx * Bxx + By * Bxy) / h1_2
                - (1.0 + Bx ** 2 + By ** 2) * 2.0 * Bx * Bxx / h1_2 ** 2)
    dh2_2_dy = (2.0 * (Bx * Bxy + By * Byy) / h1_2
                - (1.0 + Bx ** 2 + By ** 2) * 2.0 * Bx * Bxy / h1_2 ** 2)
    dh2_dx = dh2_2_dx / (2.0 * h2)                 # d_1 h_(2)
    dh2_dy = dh2_2_dy / (2.0 * h2)                 # d_2 h_(2)

    G1_11 = dh1_dx / h1                            # Gam^1_11
    G1_12 = dh1_dy / h1                            # Gam^1_12
    G1_22 = -h2 / h1_2 * dh2_dx                    # Gam^1_22
    G2_11 = -h1 / h2_2 * dh1_dy                    # Gam^2_11
    G2_12 = dh2_dx / h2                            # Gam^2_12
    G2_22 = dh2_dy / h2                            # Gam^2_22

    return Geo(B, Bx, By, J, N, N1, N2,
               g11, g22, g12,
               G1_11, G1_12, G1_22,
               G2_11, G2_12, G2_22)


# ============================================================================
# 3. 初值（Eq.(27)）
# ============================================================================
def h_initial(x, y):
    r"""Eq.(27): h = 2 当 (x-xc)^2 + (y-yc)^2 <= R0^2，否则 1。"""
    inside = ((x - XC) ** 2 + (y - YC) ** 2) <= R0 ** 2
    return torch.where(inside, torch.full_like(x, H_IC), torch.full_like(x, L_IC))


# ============================================================================
# 4. N-LAAF 网络与拟设（论文 Eq.(12)(13) + Jagtap2020）
# ============================================================================
class NLAFNet(torch.nn.Module):
    r"""
    Jagtap2020 的 neuron-wise locally adaptive activation function

        sigma( n * a_i^k * ( W^k z^{k-1} + b^k )_i ),      k = 1..D-1

    每个隐藏层的**每个神经元**各有一个可训练斜率 a_i^k（这里是长度 = 宽度的向量），
    n 为固定缩放因子；初始化满足 n a_i^k = 1（原文 "the initialization of scalable
    parameters is done such that n a_i^k = 1"），故 a_i^k = 1/n。

    slope recovery 项（原文 Eq.(2) 下方，注意**倒数**）：

        S(a) = 1 / [ (1/(D-1)) sum_{k=1}^{D-1} exp( (1/N_k) sum_i a_i^k ) ]

    输出 (h, u^1, u^2)：第 1 个输出**直接就是法向水深 h**，即方程所求的未知量。
    本算例（溃坝）没有静水表面，Eq.(13) 的参考水位 H_0 无动力学意义、只剩一个
    空壳常数（论文 §3.5："a fixed reference level of no dynamical significance
    … the ansatz is then inert"），故**不使用拟设**。
    """

    def __init__(self, width=WIDTH, layers=LAYERS, n=N_SCALE):
        super().__init__()
        self.n = float(n)
        self.hids = torch.nn.ModuleList()
        self.slopes = torch.nn.ParameterList()
        d_in = 3
        for _ in range(layers):
            lin = torch.nn.Linear(d_in, width)
            # 增益与激活匹配：PyTorch 对 tanh 的推荐增益为 5/3
            torch.nn.init.kaiming_normal_(lin.weight, nonlinearity="tanh")
            torch.nn.init.zeros_(lin.bias)
            self.hids.append(lin)
            self.slopes.append(torch.nn.Parameter(torch.full((width,), 1.0 / self.n)))
            d_in = width
        self.out = torch.nn.Linear(width, 3)
        torch.nn.init.xavier_uniform_(self.out.weight)
        torch.nn.init.zeros_(self.out.bias)

    def hidden_forward(self, X):
        # 输入归一化：以域中心为原点、半宽为尺度，映射到 [-1,1]^3
        z = torch.stack([
            (X[:, 0] - XC) / (0.5 * (X1 - X0)),
            (X[:, 1] - YC) / (0.5 * (Y1 - Y0)),
            X[:, 2] / T_END * 2.0 - 1.0,
        ], dim=1)
        for lin, a in zip(self.hids, self.slopes):
            z = torch.tanh(self.n * a * lin(z))        # N-LAAF
        return z

    def forward(self, X):
        r = self.out(self.hidden_forward(X))
        h = r[:, 0]                                    # 法向水深 h（直接输出）
        u1 = r[:, 1]
        u2 = r[:, 2]
        return h, u1, u2

    def slope_recovery(self):
        r"""S(a) = 1 / [ (1/(D-1)) sum_k exp( mean_i a_i^k ) ]（Jagtap2020，倒数形式）。"""
        Dm1 = len(self.slopes)
        denom = sum(torch.exp(a.mean()) / Dm1 for a in self.slopes)
        return 1.0 / denom


# ============================================================================
# 5. 通量 + Christoffel 收缩 + 几何源项（论文 Eq.(15) 的 hybrid 装配）
#    *** 与成功脚本 flux_and_source 完全相同 ***
# ============================================================================
def flux_and_source(h, u1, u2, geo):
    r"""
      F^{ij} = h u^i u^j + p g^{ij} ,   p = (g h^2/2) gamma      [Eq.(15) 括号项]
             (= q^i u^j + p g^{ij}，因 q^i = h u^i)
      Ch^j   = Gam^j_ik F^{ik}                                   [Eq.(15) 的 div_G 张量部分]
      S^j    = (g h^2/2) g^{ji} d_i gamma + g h g^{ji} d_i z     [Eq.(15) 后两项]
             = (g h^2/2) (grad_G gamma)^j + g h (grad_G x^3)^j    [Eq.(7)]
    正交度量 g^{12} ≡ 0（由 G_sw^{-1} 对角保证，非近似），故无交叉项残留。
    """
    q1 = h * u1
    q2 = h * u2

    p = 0.5 * G_GRAV * h ** 2 * geo.N                  # 压力标量 p = (g h^2/2) gamma

    F11 = q1 * u1 + p * geo.g11                        # hybrid：用 u 直接算，不出现 1/h
    F12 = q1 * u2 + p * geo.g12
    F22 = q2 * u2 + p * geo.g22

    Ch1 = geo.G1_11 * F11 + 2.0 * geo.G1_12 * F12 + geo.G1_22 * F22
    Ch2 = geo.G2_11 * F11 + 2.0 * geo.G2_12 * F12 + geo.G2_22 * F22

    S1 = 0.5 * G_GRAV * h ** 2 * (geo.g11 * geo.N1) + G_GRAV * h * (geo.g11 * geo.Bx)
    S2 = 0.5 * G_GRAV * h ** 2 * (geo.g22 * geo.N2) + G_GRAV * h * (geo.g22 * geo.By)

    return F11, F12, F22, Ch1, Ch2, S1, S2


# ============================================================================
# 6. PDE 残差（论文 Eq.(14)(15)）
#    *** 装配行与成功脚本 residuals 逐字相同 ***
#    *** 唯一差别：额外返回 u1, u2（供干湿罚使用），装配本身未改 ***
# ============================================================================
def residuals(X, net):
    r"""
    R1 = d_t h + (1/J)[ d_1(J q^1) + d_2(J q^2) ]                      [Eq.(14)]
    R2 = d_t q^1 + (1/J)[ d_1(J F^11) + d_2(J F^12) ] + Ch^1 + S^1     [Eq.(15)]
    R3 = d_t q^2 + (1/J)[ d_1(J F^12) + d_2(J F^22) ] + Ch^2 + S^2     [Eq.(15)]

    与 Eq.(7) 的 div_G v = (1/J) d_i(J v^i) 及
              div_G T = (1/J) d_i(J T^{ij}) + Gam^j_ik T^{ik}
    完全一致。J 与 t 无关（静态床），故 (1/J) d_t(J h) = d_t h。
    """
    X = X.clone().requires_grad_(True)
    h, u1, u2 = net(X)
    geo = geometry(X[:, 0], X[:, 1])
    J = geo.J

    q1, q2 = h * u1, h * u2
    F11, F12, F22, Ch1, Ch2, S1, S2 = flux_and_source(h, u1, u2, geo)

    Jq1, Jq2 = J * q1, J * q2
    JF11, JF12, JF22 = J * F11, J * F12, J * F22

    def grad_of(v):
        return torch.autograd.grad(v.sum(), X, create_graph=True)[0]
    # 分量约定: [0] = d_1, [1] = d_2, [2] = d_t

    g_q1 = grad_of(Jq1)
    g_q2 = grad_of(Jq2)
    g_F11 = grad_of(JF11)
    g_F12 = grad_of(JF12)
    g_F22 = grad_of(JF22)
    g_h = grad_of(J * h)

    R1 = (g_h[:, 2] + g_q1[:, 0] + g_q2[:, 1]) / J
    R2 = (g_q1[:, 2] + g_F11[:, 0] + g_F12[:, 1]) / J + Ch1 + S1
    R3 = (g_q2[:, 2] + g_F12[:, 0] + g_F22[:, 1]) / J + Ch2 + S2

    return R1, R2, R3, h, u1, u2, geo


# ============================================================================
# 7. 复合损失（论文 Eq.(16) + Jagtap2020 的 slope recovery 项）
# ============================================================================
def boundary_terms(Xb1, Xb2, net):
    r"""边界罚 L_bc（论文 Eq.(16) 的 B[U]）。

    wall  : 固壁（面内法向流量为零）。面 s^1 = const 是曲面 Σ_B 上的曲线，其
            **面内**单位法向（conormal）为  nu = t_1 / h_(1)（因 t_1 = d x / d s^1
            且 |t_1| = h_(1)）。由 q = q^1 t_1 + q^2 t_2、t_1.t_1 = h_(1)^2、
            t_1.t_2 = 0 得
                q . nu = q^i t_i . t_1 / h_(1) = q^1 h_(1)^2 / h_(1) = q^1 h_(1) ,
            故  q . nu = 0  <=>  q^1 = 0（h_(1) > 0）。同理面 s^2 = const 上要求
            q^2 = 0。这正是使 dM/dt = -∮ J q^i n_i ds = 0 的条件（见审计 A7 与
            AUDIT.md 第 3 节）。
    zgrad : 三个网络输出沿法向的一阶导数为零（Neumann），即 d_1(.) = 0 / d_2(.) = 0。
    """
    if BC_MODE == "wall":
        h1b, u1_b, _ = net(Xb1)
        h2b, _, u2_b = net(Xb2)
        q1 = h1b * u1_b                                # 面 s^1 = const：q^1 = 0
        q2 = h2b * u2_b                                # 面 s^2 = const：q^2 = 0
        return 0.5 * (torch.mean(q1 ** 2) + torch.mean(q2 ** 2))

    if BC_MODE == "zgrad":
        tot = None
        for Xb, comp in ((Xb1, 0), (Xb2, 1)):
            Xg = Xb.clone().requires_grad_(True)
            h, u1, u2 = net(Xg)
            for f in (h, u1, u2):
                d = torch.autograd.grad(f.sum(), Xg, create_graph=True)[0][:, comp]
                t = torch.mean(d ** 2)
                tot = t if tot is None else tot + t
        return tot / 6.0

    if BC_MODE == "periodic":                                  # <<< REFORM v1 (B1) >>>
        # 周期（四面粘合），对齐 Tian2025 的 dCDB9。
        #
        # 当 x^1 = X0 与 x^1 = X1 两条边在**诱导度量**下等距时，把两者等同得到的
        # 商空间（平环面）上的周期条件就是
        #     Phi(X0, x^2, t) = Phi(X1, x^2, t),      Phi in {h, u^1, u^2},
        # 同理 x^2 方向。make_points 让两对面的 (s, t) 采样点**逐点配对**，
        # 故这里直接取 Xb1 的前半 / 后半做差。
        #
        # 数学前提：粘合要求该边是等距（度量张量在缝合线上连续）。平床
        # （z == 0）上成立（平面环面）；曲床上 z_{s^1} 在两端不同（如 p1 上
        # z_1(0) 与 z_1(10) 不相等），左右边不等距，粘合后度量在缝合线上跳跃
        # —— 数学上不合法，故 apply_cli 对非平床做了硬拒绝。
        half1 = Xb1.shape[0] // 2
        hA, u1A, u2A = net(Xb1[:half1])
        hB, u1B, u2B = net(Xb1[half1:])
        t1 = (hA - hB) ** 2 + (u1A - u1B) ** 2 + (u2A - u2B) ** 2
        half2 = Xb2.shape[0] // 2
        hC, u1C, u2C = net(Xb2[:half2])
        hD, u1D, u2D = net(Xb2[half2:])
        t2 = (hC - hD) ** 2 + (u1C - u1D) ** 2 + (u2C - u2D) ** 2
        return 0.5 * (torch.mean(t1) + torch.mean(t2))

    raise ValueError("BC_MODE 必须是 'wall' / 'zgrad' / 'periodic'")



def total_loss(net, Xp, Xi, Xb1, Xb2):
    r"""
    L = w_f L_pde + w_I L_ic + w_b L_bc + w_pos L_pos + w_a S(a)   [Eq.(16)+Jagtap2020]

      L_pde = mean(R1^2 + R2^2 + R3^2)
      L_ic  = mean( (h - h^ic)^2 + (u^1)^2 + (u^2)^2 )    @ t = 0      [Eq.(16)]
      L_bc  = wall 或 zgrad 罚                              @ dU         [Eq.(16)]
      L_pos = mean(relu(-h)^2)  +  w_dw(h) |u|^2           （非负性 + 干湿一致性）
      S(a)  = slope recovery（Jagtap2020）
    """
    # ---- 残差 ----
    R1, R2, R3, h_p, u1_p, u2_p, geo_p = residuals(Xp, net)
    # <<< REFORM v1 (C2) >>> 质量方程权重 10（对齐 Tian2025 dCDB9 的 pde 权重
    #   [10,1,1,1,1,1]）：R1（质量）与 R2/R3（动量）量纲不同（[h/t] vs [h*u/t]），
    #   Tian 以 10:1 加权，此处照做。W_PDE_M 可用常量直接改。
    L_pde = torch.mean(W_PDE_M * R1 ** 2 + R2 ** 2 + R3 ** 2)


    # ---- 初值 @ t = 0：Eq.(27) 的间断台阶 ----
    h_i, u1_i, u2_i = net(Xi)
    h_ic = h_initial(Xi[:, 0], Xi[:, 1])
    L_ic = torch.mean((h_i - h_ic) ** 2 + u1_i ** 2 + u2_i ** 2)

    # ---- 边界 @ dU ----
    L_bc = boundary_terms(Xb1, Xb2, net)

    # ---- 非负性 + 干湿一致性 ----
    # 非负性（论文 Eq.(16) 的 L_pos）：
    neg = (torch.mean(torch.clamp(-h_p, min=0.0) ** 2)
           + torch.mean(torch.clamp(-h_i, min=0.0) ** 2))
    # 干湿一致性罚：水深 < H_DW 处速度必须消失（论文 §3.1：a vanishing depth implies
    # a vanishing velocity，用罚函数处理）。权重取
    #     w(h) = clamp( relu(H_DW - h) / H_DW , 0, 1 )  in [0, 1] ,
    #   h >= H_DW（湿） : w = 0，不罚；
    #   0 <= h < H_DW   : w 由 1 线性降到 0（保留原实现的平方）；
    #   h < 0（非物理） : w = 1（**封顶**）—— h < 0 的不物理程度已由非负性罚项
    #                     relu(-h)^2 单独度量，干湿项不重复计量。若不封顶，以
    #                     H_DW = 1e-2 归一化的相对权重在 h = -1 处会达到
    #                     (1.01/0.01)^2 = 10201 倍，把"轻度一致性罚"变成主导损失的量。
    #   （原拟设下 h ≈ 1 >= 0，该项恒为 0；删去参考水位后 h 由网络
    #     直接输出、初始可为负，故此封顶成为必需。）
    _wdw = torch.clamp(torch.clamp(H_DW - h_p, min=0.0) / H_DW, max=1.0)
    dw = torch.mean(_wdw ** 2 * (u1_p ** 2 + u2_p ** 2))
    L_pos = neg + dw

    # ---- slope recovery（Jagtap2020）----
    L_a = net.slope_recovery()

    L_total = (W_F * L_pde + W_I * L_ic + W_B * L_bc
               + W_POS * L_pos + W_A * L_a)
    return L_total, L_pde, L_ic, L_bc, L_pos, L_a


# ============================================================================
# 8. 配点
# ============================================================================
def _lowdisc(n, d, seed):
    r"""CPU 上的 d 维低差异序列（Sobol, scramble），返回 (n, d) 的 float64 张量。

    **在 CPU 上生成再搬到 DEVICE** —— CUDA 与 CPU 的 RNG 实现不同，若直接在
    DEVICE 上调 torch.rand 会让 cpu / cuda 得到不同的配点，破坏
    DEVICE_EQUIV_probe.log 里已核验的设备等价性。用 numpy/scipy 生成则两者
    逐位相同。
    """
    n = int(n)
    d = int(d)
    try:
        from scipy.stats import qmc
        u = qmc.Sobol(d=d, scramble=True, seed=int(seed)).random(n)
    except Exception:                      # scipy 过旧时退回固定种子伪随机
        u = np.random.RandomState(int(seed)).random_sample((n, d))
    u = np.ascontiguousarray(u, dtype=np.float64)
    with torch.device("cpu"):
        T = torch.tensor(u, dtype=torch.float64)
    return T.to(DEVICE)


def _in_box(u, three_d=True):
    r"""[0,1]^d 上的点 -> 物理域：x in [X0,X1]、y in [Y0,Y1]、t in [0,T]。"""
    x = X0 + (X1 - X0) * u[:, 0]
    y = Y0 + (Y1 - Y0) * u[:, 1]
    if three_d:
        t = T_END * u[:, 2]
    else:
        t = torch.zeros_like(x)
    return torch.stack([x, y, t], dim=1)


def snap_times(step=0.1):
    r"""快照时刻表。

    T_SNAP 为 None（默认）时自动取 step 的整数倍（**含 0**）直到 T_END
    （T_END = 1.0 -> [0.0, 0.1, ..., 1.0]，共 11 帧）；否则沿用 T_SNAP 中落在
    [0, T_END] 里的时刻。
    """
    if T_SNAP is not None:
        return [float(t) for t in T_SNAP if 0.0 <= t <= T_END + 1e-12]
    n = int(np.floor(T_END / step + 1e-9))
    return [round(k * step, 10) for k in range(0, n + 1)]   # <<< FIX1 (P2-2) >>>


def _check_in_domain(X, tag=""):
    r"""域包含性硬检查：所有配点必须落在 U x [0,T] 内，否则抛异常。

    此不变量来自成功脚本的 D1 缺陷（四条边界曾共用同一参数化，使竖边上 45% 的点
    越域），这里写成可执行断言防止复发。
    """
    out = ((X[:, 0] < X0 - 1e-12) | (X[:, 0] > X1 + 1e-12)
           | (X[:, 1] < Y0 - 1e-12) | (X[:, 1] > Y1 + 1e-12)
           | (X[:, 2] < 0.0) | (X[:, 2] > T_END + 1e-12))
    n_out = int(out.sum())
    if n_out:
        raise AssertionError(
            f"[域检查] {tag}: {n_out}/{X.shape[0]} 个配点落在 U x [0,T] 之外")
    return True


def make_points(nx=None, ny=None, nt=None, m=None, sampler=None):
    r"""配点。

    sampler == "sobol"（默认，消除时域混叠）：
        Xp  : N_INT   个 3D 低差异点（域 U x [0,T]）
        Xi  : N_IC    个 2D 低差异点（初始面 t = 0）
        Xb1 : 2*N_BC_SIDE 个点（面 s^1 = X0 与 s^1 = X1）
        Xb2 : 2*N_BC_SIDE 个点（面 s^2 = Y0 与 s^2 = Y1）
        Xb  : Xb1 与 Xb2 的拼接（供统一审计）

        两对面的第 i 个点被构造成具有**相同的面内坐标与相同的 t**（配对），
        这是周期边界（--bc periodic）所必需的：周期罚直接取前后两半之差。

    sampler == "grid"：旧版张量网格（逐字保留，供 A/B 对照）。
    """
    sampler = SAMPLER if sampler is None else str(sampler)

    if sampler == "grid":
        nx = NX if nx is None else nx
        ny = NY if ny is None else ny
        nt = NT if nt is None else nt
        m = M_BC if m is None else m

        xv = torch.linspace(X0, X1, nx, dtype=torch.float64)
        yv = torch.linspace(Y0, Y1, ny, dtype=torch.float64)
        tv = torch.linspace(0.0, T_END, nt, dtype=torch.float64)

        Xg, Yg, Tg = torch.meshgrid(xv, yv, tv, indexing="ij")
        Xp = torch.stack([Xg, Yg, Tg], dim=-1).reshape(-1, 3)
        Xi = torch.stack([Xg[..., 0], Yg[..., 0], torch.zeros_like(Tg[..., 0])],
                         dim=-1).reshape(-1, 3)

        # 两类边各用自己的长度参数，绝不共用（AUDIT.md D1）
        ex = torch.linspace(X0, X1, m, dtype=torch.float64)
        ey = torch.linspace(Y0, Y1, m, dtype=torch.float64)
        tb = torch.linspace(0.0, T_END, nt, dtype=torch.float64)

        def with_t(xy):
            return torch.cat([xy.repeat_interleave(nt, dim=0),
                              tb.repeat(xy.shape[0]).unsqueeze(-1)], dim=1)

        x_lo = torch.full((m,), X0, dtype=torch.float64)
        x_hi = torch.full((m,), X1, dtype=torch.float64)
        y_lo = torch.full((m,), Y0, dtype=torch.float64)
        y_hi = torch.full((m,), Y1, dtype=torch.float64)
        Xb1 = with_t(torch.cat([torch.stack([x_lo, ey], dim=-1),
                                torch.stack([x_hi, ey], dim=-1)], dim=0))
        Xb2 = with_t(torch.cat([torch.stack([ex, y_lo], dim=-1),
                                torch.stack([ex, y_hi], dim=-1)], dim=0))
        Xb = torch.cat([Xb1, Xb2], dim=0)

        _check_in_domain(Xp, "内部配点")
        _check_in_domain(Xi, "初始配点")
        _check_in_domain(Xb1, "边界配点 s1 面")
        _check_in_domain(Xb2, "边界配点 s2 面")
        return Xp, Xi, Xb1, Xb2, Xb

    # ---- 3D 低差异采样（默认）----
    Xp = _in_box(_lowdisc(N_INT, 3, SEED + 101), three_d=True)
    Xi = _in_box(_lowdisc(N_IC, 2, SEED + 102), three_d=False)

    u1 = _lowdisc(N_BC_SIDE, 2, SEED + 111)     # (s, t) 供 x 方向的两个面
    u2 = _lowdisc(N_BC_SIDE, 2, SEED + 112)     # (s, t) 供 y 方向的两个面

    def _edge(u, fixed, axis):
        s = u[:, 0]
        tb = T_END * u[:, 1]
        if axis == 0:                            # x^1 固定，s -> x^2
            x = torch.full_like(s, float(fixed))
            y = Y0 + (Y1 - Y0) * s
        else:                                    # x^2 固定，s -> x^1
            x = X0 + (X1 - X0) * s
            y = torch.full_like(s, float(fixed))
        return torch.stack([x, y, tb], dim=1)

    Xb1 = torch.cat([_edge(u1, X0, 0), _edge(u1, X1, 0)], dim=0)
    Xb2 = torch.cat([_edge(u2, Y0, 1), _edge(u2, Y1, 1)], dim=0)
    Xb = torch.cat([Xb1, Xb2], dim=0)

    _check_in_domain(Xp, "内部配点")
    _check_in_domain(Xi, "初始配点")
    _check_in_domain(Xb1, "边界配点 s1 面")
    _check_in_domain(Xb2, "边界配点 s2 面")
    return Xp, Xi, Xb1, Xb2, Xb


# ============================================================================
# 9. 审计（不参与训练）—— 内蕴几何与度量感知算子的硬证据
# ============================================================================
def _mx(v):
    return float(v.detach().abs().max())


def _grad(v, X):
    return torch.autograd.grad(v.sum(), X, create_graph=True)[0]


def audit_geometry_and_operators(X, tag=""):
    r"""在给定配点上核验几何层与度量化算子的恒等式（全部对解析场用 autograd 求导）：

      (G1) gamma * J = 1                                          [Eq.(4)(5)]
      (G2) gamma = 1/sqrt(1+|grad z|^2)                           [Eq.(2)(3)->(5)]
      (G3) 逆度量互逆：g^{11} h_(1)^2 = 1, g^{22} h_(2)^2 = 1, g^{12} = 0   [Eq.(3)]
      (G4) 六个 Christoffel 由度量**独立重构**后逐项相等：
             Gam^j_ik = (1/2) g^{jl}(d_i G_kl + d_k G_il - d_l G_ik)
           其中 G_kl 由标架范数给出（对角），其偏导用 autograd 得到。
      (G5) Christoffel 迹恒等式 Gam^j_{ji} = d_i ln J（真实的几何恒等式，
           与 G4 独立：左边用解析 Christoffel，右边用 autograd 求 ln J 的导数）
      (G6) 向量散度： (1/J) d_i(J v^i) 与 Christoffel 显式展开一致
      (O1) 静水态 zeta = const, q = 0 时 R2 = R3 = 0（算子精确湮灭 ISWE 的真解）
      (O2) 良平衡强形式（任意光滑正水深）：q = 0 时 (div_G T + S)^j = g h g^{ji} d_i zeta
    """
    X = X.clone().requires_grad_(True)
    geo = geometry(X[:, 0], X[:, 1])

    # ---- (G1)(G2)(G3) ----
    e1 = _mx(geo.N * geo.J - 1.0)
    gz = _grad(geo.B, X)
    e2 = _mx(geo.N - 1.0 / torch.sqrt(1.0 + gz[:, 0] ** 2 + gz[:, 1] ** 2))
    e3 = _mx(torch.stack([
        geo.g11 * (1.0 + geo.Bx ** 2) - 1.0,
        geo.g22 * (1.0 + geo.Bx ** 2 + geo.By ** 2) / (1.0 + geo.Bx ** 2) - 1.0,
        geo.g12], dim=1))

    # ---- (G4) 由度量独立重构 Christoffel ----
    G11 = 1.0 + geo.Bx ** 2                                        # h_(1)^2
    G22 = (1.0 + geo.Bx ** 2 + geo.By ** 2) / (1.0 + geo.Bx ** 2)  # h_(2)^2
    G12 = torch.zeros_like(G11)
    d1G11, d2G11 = _grad(G11, X)[:, 0], _grad(G11, X)[:, 1]
    d1G22, d2G22 = _grad(G22, X)[:, 0], _grad(G22, X)[:, 1]
    ig11, ig22 = 1.0 / G11, 1.0 / G22                              # g^{11}, g^{22}
    # 标准式（对角度量 + G12 恒 0，故只剩下列分量非零）
    C1_11 = 0.5 * ig11 * (d1G11 + d1G11 - d1G11)
    C1_12 = 0.5 * ig11 * (d2G11 + 0.0 - 0.0)
    C1_22 = 0.5 * ig11 * (0.0 + 0.0 - d1G22)
    C2_11 = 0.5 * ig22 * (0.0 + 0.0 - d2G11)
    C2_12 = 0.5 * ig22 * (d1G22 + 0.0 - 0.0)
    C2_22 = 0.5 * ig22 * (d2G22 + d2G22 - d2G22)
    e4 = _mx(torch.stack([geo.G1_11 - C1_11, geo.G1_12 - C1_12,
                          geo.G1_22 - C1_22, geo.G2_11 - C2_11,
                          geo.G2_12 - C2_12, geo.G2_22 - C2_22], dim=1))

    # ---- (G5) 迹恒等式 Gam^j_{ji} = d_i ln J ----
    d1lnJ = _grad(torch.log(geo.J), X)[:, 0]
    d2lnJ = _grad(torch.log(geo.J), X)[:, 1]
    e5 = _mx(torch.stack([(geo.G1_11 + geo.G2_12) - d1lnJ,
                          (geo.G1_12 + geo.G2_22) - d2lnJ], dim=1))

    # ---- (G6) 向量散度 ----
    v1 = X[:, 0] ** 2 * torch.sin(X[:, 1])
    v2 = X[:, 1] * torch.cos(X[:, 0])
    div_v = (_grad(geo.J * v1, X)[:, 0] + _grad(geo.J * v2, X)[:, 1]) / geo.J
    d1v1 = _grad(v1, X)[:, 0]
    d2v2 = _grad(v2, X)[:, 1]
    div_v_ref = (d1v1 + d2v2
                 + v1 * (geo.G1_11 + geo.G2_12)      # = d_1 ln J
                 + v2 * (geo.G1_12 + geo.G2_22))     # = d_2 ln J
    e6 = _mx(div_v - div_v_ref)

    # ---- (O1) 静水态：取 zeta = max(z) + 1 保证 h > 0 ----
    zeta_test = float(geo.B.detach().max()) + 1.0
    h_rest = (zeta_test - geo.B) / geo.N
    u0 = torch.zeros_like(h_rest)
    F11, F12, F22, Ch1, Ch2, S1, S2 = flux_and_source(h_rest, u0, u0, geo)
    R2r = (_grad(geo.J * F11, X)[:, 0] + _grad(geo.J * F12, X)[:, 1]) / geo.J + Ch1 + S1
    R3r = (_grad(geo.J * F12, X)[:, 0] + _grad(geo.J * F22, X)[:, 1]) / geo.J + Ch2 + S2
    e7 = max(_mx(R2r), _mx(R3r))

    # ---- (O2) 良平衡强形式，取任意光滑正水深 h（非平凡，且与任何参考水位无关）----
    #   恒等式（对**任意** h 成立，q = 0）：
    #       (1/J) d_i(J F^{1i}) + Ch^1 + S^1 = g h g^{1i} d_i zeta ,   zeta = z + gamma h
    #   证明：T^{ij} = p g^{ij}，p = (g h^2/2) gamma；由度量相容性
    #       (1/J) d_i(J T^{ji}) + Ch^j = g^{ji} d_i p ,
    #   故左端
    #       = g^{ji} d_i p + (g h^2/2) g^{ji} d_i gamma + g h g^{ji} d_i z
    #       = g h g^{ji} ( gamma d_i h + h d_i gamma + d_i z ) = g h g^{ji} d_i zeta 。
    #   这里 zeta 与右端都由几何量直接算出，**不经过**任何参考水位。
    h_t2 = (1.0 + 0.137 * torch.sin(torch.pi * (X[:, 0] - X0) / (X1 - X0))
            * torch.cos(2.0 * torch.pi * (X[:, 1] - Y0) / (Y1 - Y0)))
    F11b, F12b, F22b, Ch1b, Ch2b, S1b, S2b = flux_and_source(h_t2, u0, u0, geo)
    Rb2 = (_grad(geo.J * F11b, X)[:, 0] + _grad(geo.J * F12b, X)[:, 1]) / geo.J + Ch1b + S1b
    Rb3 = (_grad(geo.J * F12b, X)[:, 0] + _grad(geo.J * F22b, X)[:, 1]) / geo.J + Ch2b + S2b
    zeta_t = geo.B + geo.N * h_t2
    dzeta = _grad(zeta_t, X)
    rhs2 = G_GRAV * h_t2 * (geo.g11 * dzeta[:, 0] + geo.g12 * dzeta[:, 1])
    rhs3 = G_GRAV * h_t2 * (geo.g12 * dzeta[:, 0] + geo.g22 * dzeta[:, 1])
    e8 = max(_mx(Rb2 - rhs2), _mx(Rb3 - rhs3))

    print(f"  [审计:{tag}] 床 = {BED}  域 [{X0:g},{X1:g}]x[{Y0:g},{Y1:g}]  "
          f"R0 = {R0:g}")
    print(f"  [审计:{tag}] (G1) |gamma J - 1|                      = {e1:.3e}")
    print(f"  [审计:{tag}] (G2) |gamma - 1/sqrt(1+|grad z|^2)|     = {e2:.3e}")
    print(f"  [审计:{tag}] (G3) 逆度量互逆（g11,g22,g12）          = {e3:.3e}")
    print(f"  [审计:{tag}] (G4) 6 个 Christoffel vs 度量独立重构    = {e4:.3e}")
    print(f"  [审计:{tag}] (G5) 迹恒等式 Gam^j_ji = d_i ln J         = {e5:.3e}")
    print(f"  [审计:{tag}] (G6) 向量散度 vs Christoffel 显式展开     = {e6:.3e}")
    print(f"  [审计:{tag}] (O1) 静水态 max|R2|, max|R3|              = {e7:.3e}")
    print(f"  [审计:{tag}] (O2) 良平衡强形式(任意 h)                 = {e8:.3e}")
    # ---- (F1) 坐标口径自检                                     # <<< REFORM v1 (F1) >>>
    #   代码一律用 d/dx^i；论文 (2.3a)(2.3b) 是 d/ds^1 = d/dx^1 而
    #       d/ds^2 = d/dx^2 - c d/dx^1,   c = z_{s^1} z_{s^2} / (1 + z_{s^1}^2)。
    #   故**仅当 z_{s^1} z_{s^2} == 0** 时两套口径逐点相同。本脚本的 flat 与 p1
    #   都满足；p2 / p3 不满足，会引入 O(c) 偏差（实测对照床 z = x^2/25 + y^2/9
    #   上 c ~ 0.18）。把这条写成可执行检查，换床时不会静默出错。
    cross_zz = _mx(geo.Bx * geo.By)
    print(f"  [审计:{tag}] (F1) max|z_1 z_2|（坐标口径重合判据）    = {cross_zz:.3e}")

    tol = 1e-9
    ok = (e1 < 1e-12 and e2 < 1e-12 and e3 < 1e-12
          and e4 < tol and e5 < tol and e6 < tol and e7 < tol and e8 < tol)
    # <<< REFORM v1 (F1) >>> flat / p1 上 z_{s^1} z_{s^2} 必须恒为 0；否则拒绝。
    if BED in ("flat", "p1"):
        ok = ok and (cross_zz < 1e-12)

    print(f"  [审计:{tag}] 几何与算子{'全部通过 ✅' if ok else '存在失败 ❌'}")
    if not ok:
        raise AssertionError(
            f"[审计:{tag}] 几何/算子恒等式未通过: G1={e1:.3e} G2={e2:.3e} "
            f"G3={e3:.3e} G4={e4:.3e} G5={e5:.3e} G6={e6:.3e} "
            f"O1={e7:.3e} O2={e8:.3e}")
    return ok


def audit_classical_reduction(nn=7):
    r"""退化检验：零床上 gamma = 1, J = 1, g^{ij} = delta^{ij}, Christoffel = 0，
    内蕴残差 Eq.(14)(15) 必须逐项退化为经典 SWE：

        h_t + (h u)_x + (h v)_y = 0
        (h u)_t + (h u^2 + g h^2/2)_x + (h u v)_y = 0
        (h v)_t + (h u v)_x + (h v^2 + g h^2/2)_y = 0

    在一个**任意光滑场**上把两条独立路径逐点相减，不依赖读代码。
    """
    global BED
    saved = BED
    BED = "flat"
    try:
        xv = torch.linspace(X0 + 0.3, X1 - 0.3, nn, dtype=torch.float64)
        yv = torch.linspace(Y0 + 0.3, Y1 - 0.3, nn, dtype=torch.float64)
        Xg, Yg = torch.meshgrid(xv, yv, indexing="ij")
        X = torch.stack([Xg.reshape(-1), Yg.reshape(-1),
                         torch.full((nn * nn,), 0.4, dtype=torch.float64)],
                        dim=-1).requires_grad_(True)
        xx, yy = X[:, 0], X[:, 1]

        h = 1.3 + 0.2 * torch.sin(xx) * torch.cos(2.0 * yy)
        u1 = 0.3 * torch.cos(xx + yy)
        u2 = -0.2 * torch.sin(2.0 * xx - yy)
        geo = geometry(xx, yy)

        q1, q2 = h * u1, h * u2
        F11, F12, F22, Ch1, Ch2, S1, S2 = flux_and_source(h, u1, u2, geo)
        R1i = (_grad(geo.J * h, X)[:, 2] + _grad(geo.J * q1, X)[:, 0]
               + _grad(geo.J * q2, X)[:, 1]) / geo.J
        R2i = (_grad(geo.J * q1, X)[:, 2] + _grad(geo.J * F11, X)[:, 0]
               + _grad(geo.J * F12, X)[:, 1]) / geo.J + Ch1 + S1
        R3i = (_grad(geo.J * q2, X)[:, 2] + _grad(geo.J * F12, X)[:, 0]
               + _grad(geo.J * F22, X)[:, 1]) / geo.J + Ch2 + S2

        dt = lambda v: _grad(v, X)[:, 2]      # noqa: E731
        dx = lambda v: _grad(v, X)[:, 0]      # noqa: E731
        dy = lambda v: _grad(v, X)[:, 1]      # noqa: E731
        Rc1 = dt(h) + dx(q1) + dy(q2)
        Rc2 = dt(q1) + dx(q1 * u1 + 0.5 * G_GRAV * h ** 2) + dy(q1 * u2)
        Rc3 = dt(q2) + dx(q1 * u2) + dy(q2 * u2 + 0.5 * G_GRAV * h ** 2)

        e1, e2, e3 = _mx(R1i - Rc1), _mx(R2i - Rc2), _mx(R3i - Rc3)
        print(f"  [审计:退化] 零床上 |R1-R1_cls|, |R2-R2_cls|, |R3-R3_cls| = "
              f"{e1:.3e}, {e2:.3e}, {e3:.3e}")
        ok = max(e1, e2, e3) < 1e-11
        print(f"  [审计:退化] 经典 SWE 退化{'通过 ✅' if ok else '失败 ❌'}")
        if not ok:
            raise AssertionError(f"[审计:退化] 未退化为经典 SWE: "
                                 f"{e1:.3e}, {e2:.3e}, {e3:.3e}")
        return ok
    finally:
        BED = saved


def audit_surface_element(n=161):
    r"""(A7) J 作为**曲面面积元**的独立核验。

    论文 Eq.(4) 断言 J = sqrt(det G_sw) 就是床面的面积元。这里用两条互不依赖的
    途径算同一个面积：

      (i)  真实三维嵌入 P_ij = (x_i, y_j, z_ij)：每个网格四边形拆成两个三角形，
           累加其三维面积。这一支**只用 z 的取值**，完全不用 J / 任何导数公式。
      (ii) 代码里的 J 做梯形求积 ∫_U J dx dy。

    两者都是同一积分的 O(dx^2) 求积，故相对差应为 O(dx^2) 且随网格加密按 4 倍下降。
    若 J 的公式写错（漏掉 sqrt(1+|grad z|^2)、误取 J = 1、或 h_(1)h_(2) 的因子
    取错），两支会被拉开 O(1) 的相对差，本检验即其探测器。
    """
    _trapz = getattr(np, "trapezoid", None)
    if _trapz is None:                                  # numpy < 2.0
        _trapz = np.trapz                                 # noqa: NPY201

    def _area(nn):
        xv = np.linspace(X0, X1, nn)
        yv = np.linspace(Y0, Y1, nn)
        Xg, Yg = np.meshgrid(xv, yv, indexing="ij")
        xt = torch.tensor(Xg.reshape(-1), dtype=torch.float64)
        yt = torch.tensor(Yg.reshape(-1), dtype=torch.float64)
        with torch.no_grad():
            geo = geometry(xt, yt)
            Zg = geo.B.cpu().numpy().reshape(nn, nn)
            Jg = geo.J.cpu().numpy().reshape(nn, nn)
        A_J = float(_trapz(_trapz(Jg, yv, axis=1), xv, axis=0))
        P = np.stack([Xg, Yg, Zg], axis=-1)
        a, b, c, d = P[:-1, :-1], P[1:, :-1], P[1:, 1:], P[:-1, 1:]

        def tri(p, q, r):
            return 0.5 * np.linalg.norm(np.cross(q - p, r - p), axis=-1)

        A_e = float(np.sum(tri(a, b, c) + tri(a, c, d)))
        return A_e, A_J

    A_e, A_J = _area(n)
    rel = abs(A_e - A_J) / max(abs(A_J), 1e-30)
    A_e2, A_J2 = _area(2 * n)
    rel2 = abs(A_e2 - A_J2) / max(abs(A_J2), 1e-30)
    print(f"  [审计:A7] 曲面面积：嵌入三角剖分 {A_e:.8f} / ∫∫ J dx dy "
          f"{A_J:.8f}（相对差 {rel:.3e}）")
    print(f"  [审计:A7] 网格加密 2 倍后相对差 = {rel2:.3e}"
          f"（O(dx^2) 期望 ~{rel / 4.0:.3e}）")
    tol = 5.0e-3
    ok = rel < tol and rel2 <= rel + 1e-12
    print(f"  [审计:A7] J 是否为真实曲面面积元（且误差随网格收敛）"
          f"{'通过 ✅' if ok else '失败 ❌'}")
    if not ok:
        raise AssertionError(f"[审计:A7] J 非曲面面积元: A_embed={A_e}, A_J={A_J}, "
                             f"rel={rel:.3e}, rel2={rel2:.3e}")
    return ok


def audit_slope_recovery():
    r"""slope recovery 闭式核验：a_i^k 初值 1/n，故 mean_i a_i^k = 1/n，
    于是 S(a) = 1/exp(1/n) = exp(-1/n)；同时核验 n*a = 1。"""
    net = NLAFNet(width=8, layers=3, n=N_SCALE)
    S = float(net.slope_recovery())
    ref = float(np.exp(-1.0 / N_SCALE))
    na = [float((N_SCALE * a).mean().detach()) for a in net.slopes]
    print(f"  [审计:N-LAAF] n = {N_SCALE:g}；初值 n*a 的层均值 = {na}")
    print(f"  [审计:N-LAAF] S(a) = {S:.12f}，闭式 exp(-1/n) = {ref:.12f}，"
          f"差 = {abs(S - ref):.3e}")
    ok = abs(S - ref) < 1e-12 and all(abs(v - 1.0) < 1e-12 for v in na)
    print(f"  [审计:N-LAAF] {'通过 ✅' if ok else '失败 ❌'}")
    if not ok:
        raise AssertionError("[审计:N-LAAF] slope recovery 或斜率初值与闭式不符")
    return ok


def audit_ic_and_points():
    r"""核验初值 Eq.(27) 与配点集：
        - h 取值恰为 {1, 2}，圆盘面积占比与 pi R0^2/|U| 一致（同一网格上）
        - 零床上初始总质量与闭式对照
        - 全部配点在域内；边界点恰好落在四个面上；竖边不越域
    """
    n = 601
    xv = torch.linspace(X0, X1, n, dtype=torch.float64)
    yv = torch.linspace(Y0, Y1, n, dtype=torch.float64)
    Xg, Yg = torch.meshgrid(xv, yv, indexing="ij")
    hv = h_initial(Xg.reshape(-1), Yg.reshape(-1))
    frac_in = float((hv > 1.5).double().mean())
    frac_ref = float(np.pi * R0 ** 2 / ((X1 - X0) * (Y1 - Y0)))
    hi, lo = float(hv.max()), float(hv.min())

    Xp, Xi, Xb1, Xb2, Xb = make_points()
    on_s1 = int((((Xb1[:, 0] - X0).abs() < 1e-12)
                 | ((Xb1[:, 0] - X1).abs() < 1e-12)).sum())
    on_s2 = int((((Xb2[:, 1] - Y0).abs() < 1e-12)
                 | ((Xb2[:, 1] - Y1).abs() < 1e-12)).sum())

    print(f"  [审计:IC] h 取值范围 = [{lo:.6f}, {hi:.6f}]（应恰为 [1, 2]）")
    print(f"  [审计:IC] 圆盘面积占比 = {frac_in:.6f}，闭式 pi R0^2/|U| = "
          f"{frac_ref:.6f}，差 = {abs(frac_in - frac_ref):.3e}（网格离散 O(1/n)）")
    print(f"  [审计:IC] 边界点落在面上: s1 面 {on_s1}/{Xb1.shape[0]}, "
          f"s2 面 {on_s2}/{Xb2.shape[0]}")
    print(f"  [审计:IC] 配点规模: Nf={Xp.shape[0]} NI={Xi.shape[0]} "
          f"Nb1={Xb1.shape[0]} Nb2={Xb2.shape[0]} 合计 {Xb.shape[0]}")

    ok = (abs(hi - H_IC) < 1e-12 and abs(lo - L_IC) < 1e-12
          and abs(frac_in - frac_ref) < 5.0 / n
          and on_s1 == Xb1.shape[0] and on_s2 == Xb2.shape[0])
    if BED == "flat":
        # 零床：J = 1，故 M = ∫_U h dx dy 有闭式。用**等权**求和（n x n 个采样点）
        # 近似积分。被积函数在圆盘边界上间断，其求积误差是 O(dx * 周长)，**不是**
        # 机器精度；故判据必须与 dx 一致，不能写死一个与分辨率无关的小数。
        dx = (X1 - X0) / (n - 1)
        dxdy = (X1 - X0) * (Y1 - Y0) / n ** 2
        M_disc = float(hv.reshape(n, n).sum()) * dxdy
        M_exact = L_IC * (X1 - X0) * (Y1 - Y0) + (H_IC - L_IC) * np.pi * R0 ** 2
        rel_M = abs(M_disc - M_exact) / M_exact
        bound_M = 4.0 * (H_IC - L_IC) * (2.0 * np.pi * R0) * dx / M_exact
        print(f"  [审计:IC] 零床初始质量: 离散 {M_disc:.6f} / 闭式 {M_exact:.6f} "
              f"（相对差 {rel_M:.3e} < 理论界 {bound_M:.3e}）")
        ok = ok and rel_M < bound_M
    print(f"  [审计:IC] {'通过 ✅' if ok else '失败 ❌'}")
    if not ok:
        raise AssertionError("[审计:IC] 初值 / 配点 / 质量检查未通过")
    return ok


def run_audits():
    print("=" * 78)
    print("  审计（不训练）：内蕴几何、度量感知算子、经典 SWE 退化、N-LAAF、初值")
    print("=" * 78)
    Xp, Xi, Xb1, Xb2, Xb = make_points()
    audit_ic_and_points()
    audit_geometry_and_operators(Xp, tag="内部配点")
    audit_geometry_and_operators(Xi, tag="初始面  ")
    audit_geometry_and_operators(Xb, tag="边界    ")
    audit_classical_reduction()
    audit_surface_element()
    audit_slope_recovery()
    print("=" * 78)
    print("  审计全部通过 ✅")
    print("=" * 78)


# ============================================================================
# 10. 梯度范数监控 / 重平衡（Wang2021 式；只动权重，不碰 ISWE 数学）
# ============================================================================
def grad_norm_report(net, Xp, Xi, Xb1, Xb2, do_rebalance):
    r"""分别计算四个损失项对参数的梯度范数（mean|grad|），并按 Wang2021 的
    learning-rate-annealing 规则给出建议权重  lambda_i = mean|grad L_total| / mean|grad L_i|。
    do_rebalance=True 时按该建议做一次指数滑动更新。返回当前权重字典。
    """
    global W_F, W_I, W_B, W_POS
    terms = {}
    (R1, R2, R3, h_p, u1_p, u2_p, geo_p) = residuals(Xp, net)
    h_i, u1_i, u2_i = net(Xi)
    h_ic = h_initial(Xi[:, 0], Xi[:, 1])
    L = {
        "pde": torch.mean(R1 ** 2 + R2 ** 2 + R3 ** 2),
        "ic": torch.mean((h_i - h_ic) ** 2 + u1_i ** 2 + u2_i ** 2),
        "bc": boundary_terms(Xb1, Xb2, net),
        "pos": (torch.mean(torch.clamp(-h_p, min=0.0) ** 2)
                + torch.mean(torch.clamp(-h_i, min=0.0) ** 2)),
    }
    gn = {}
    for k, v in L.items():
        net.zero_grad(set_to_none=True)
        v.backward(retain_graph=False)
        tot = 0.0
        cnt = 0
        for q in net.parameters():
            if q.grad is not None:
                tot += float(q.grad.abs().mean())
                cnt += 1
        gn[k] = tot / max(cnt, 1)
    net.zero_grad(set_to_none=True)
    if do_rebalance and gn["pde"] > 0:
        base = gn["pde"]
        new = {}
        for k, w in (("ic", W_I), ("bc", W_B), ("pos", W_POS)):
            if gn[k] > 0:
                lam = base / gn[k]
                new[k] = 0.5 * w + 0.5 * lam          # 指数滑动（等同动量 0.5）
        W_I = new.get("ic", W_I)
        W_B = new.get("bc", W_B)
        W_POS = new.get("pos", W_POS)
    return gn


# ============================================================================
# 11. 评估与出图
# ============================================================================
def radial_profile(h2d, xe, ye, nb=121):
    """径向平均剖面；返回 (r, h_mean)。"""
    Xg, Yg = np.meshgrid(xe, ye, indexing="ij")
    r = np.sqrt((Xg - XC) ** 2 + (Yg - YC) ** 2).ravel()
    hh = h2d.ravel()
    rb = np.linspace(0.0, 0.5 * min(X1 - X0, Y1 - Y0), nb)
    rr, hr = [], []
    for k in range(len(rb) - 1):
        m = (r >= rb[k]) & (r < rb[k + 1])
        if m.sum() > 0:
            rr.append(0.5 * (rb[k] + rb[k + 1]))
            hr.append(float(hh[m].mean()))
    return np.array(rr), np.array(hr)


def evaluate_field(net, ne_x=241, ne_y=241, t_list=None):
    r"""在细网格上评估若干时刻的场，并计算诊断量：
        总质量 M(t) = ∫ h J dx dy（固壁下精确守恒）
        激波半径 = 径向平均剖面 |dh/dr| 的极大点
    """
    if t_list is None:
        t_list = [0.0, 0.25 * T_END, 0.5 * T_END, 0.75 * T_END, T_END]
    xe = np.linspace(X0, X1, ne_x)
    ye = np.linspace(Y0, Y1, ne_y)
    Xg, Yg = np.meshgrid(xe, ye, indexing="ij")
    xy = np.stack([Xg.ravel(), Yg.ravel()], axis=1)
    area = (xe[1] - xe[0]) * (ye[1] - ye[0])
    # 梯形权重：与 Q_out 的 _trapz 同口径，避免"求积不相容"在守恒核验里
    # 制造 O(dx) 伪残差。                                # <<< FIX1 (P2-5) >>>
    _wx = np.full(xe.size, xe[1] - xe[0])
    _wx[0] *= 0.5
    _wx[-1] *= 0.5
    _wy = np.full(ye.size, ye[1] - ye[0])
    _wy[0] *= 0.5
    _wy[-1] *= 0.5
    W2 = np.outer(_wx, _wy)

    out = {"x": xe, "y": ye, "t_list": np.array(t_list),
           "h": [], "u1": [], "u2": [], "zeta": [],
           "mass": [], "shock_r": [], "r_prof": [], "h_prof": []}

    for tv in t_list:
        X = torch.tensor(np.column_stack([xy, np.full(xy.shape[0], tv)]),
                         dtype=torch.float64)
        with torch.no_grad():
            h, u1, u2 = net(X)
            geo = geometry(X[:, 0], X[:, 1])
            h_np = h.cpu().numpy().reshape(Xg.shape)
            J_np = geo.J.cpu().numpy().reshape(Xg.shape)
            zeta = (h * geo.N + geo.B).cpu().numpy().reshape(Xg.shape)
            u1_np = u1.cpu().numpy().reshape(Xg.shape)
            u2_np = u2.cpu().numpy().reshape(Xg.shape)
        rr, hr = radial_profile(h_np, xe, ye)
        shock_r = float(rr[int(np.argmax(np.abs(np.gradient(hr, rr))))]) \
            if len(rr) > 3 else float("nan")
        out["h"].append(h_np)
        out["u1"].append(u1_np)
        out["u2"].append(u2_np)
        out["zeta"].append(zeta)
        out["mass"].append(float(np.sum(h_np * J_np * W2)))   # <<< FIX1 (P2-5b) >>>
        out["shock_r"].append(shock_r)
        out["r_prof"].append(rr)
        out["h_prof"].append(hr)

    for k in ("h", "u1", "u2", "zeta"):
        out[k] = np.array(out[k])
    for k in ("mass", "shock_r"):
        out[k] = np.array(out[k])
    out["r_prof"] = np.array(out["r_prof"])
    out["h_prof"] = np.array(out["h_prof"])
    return out


def report_eval(ev, ref=None):
    print()
    print("=" * 78)
    print("  评估")
    print("=" * 78)
    M0 = ev["mass"][0]
    print("  总质量 M(t) = ∫ h J dx dy（固壁下由 Eq.(14) 精确守恒）:")
    for tv, M in zip(ev["t_list"], ev["mass"]):
        print(f"    t = {tv:5.3f} s : M = {M:.8f}   (M - M(0))/M(0) = "
              f"{(M - M0) / M0:+.3e}")
    print(f"  零床初始质量闭式         = "
          f"{L_IC * (X1 - X0) * (Y1 - Y0) + (H_IC - L_IC) * np.pi * R0 ** 2:.8f}"
          if BED == "flat" else "  (非零床：初始质量由积分给出)")
    print("  径向平均剖面的激波半径估计 r_shock = argmax |dh/dr|:")
    for tv, r in zip(ev["t_list"], ev["shock_r"]):
        print(f"    t = {tv:5.3f} s : r_shock = {r:.4f} m "
              f"(= {100.0 * r / (X1 - X0):.2f}% 域长)")
    hT = ev["h"][-1]
    print(f"  t = T 时 h 范围 = [{hT.min():.6f}, {hT.max():.6f}]")
    if ref is not None:
        err = np.abs(hT - ref["h"])
        rms = float(np.sqrt(np.mean((hT - ref["h"]) ** 2)))
        rel1 = float(err.sum() / max(np.abs(ref["h"]).sum(), 1e-30))
        print(f"  [参考解] MAE = {float(err.mean()):.6e}  RMSE = {rms:.6e}  "
              f"相对 L1 = {rel1:.6e}          [Eq.(22)]")
    else:
        print("  [参考解] 未提供（--ref）。论文 §4.6 的误差须由内蕴有限体积解给出。")
    print("=" * 78)


def make_plots(ev, outdir, hist):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 11})
    xe, ye = ev["x"], ev["y"]
    hT = ev["h"][-1]
    tag = BED

    fig, ax = plt.subplots(figsize=(7.8, 6.2))
    im = ax.pcolormesh(xe, ye, hT.T, cmap="viridis", shading="auto")
    fig.colorbar(im, ax=ax, shrink=0.9, label=r"$h_\theta$  [m]")
    th = np.linspace(0, 2 * np.pi, 400)
    ax.plot(XC + R0 * np.cos(th), YC + R0 * np.sin(th), "w--", lw=1.3,
            label=rf"initial dam radius $R_0={R0:g}$ m")
    ax.set_xlabel(r"$x^1$")
    ax.set_ylabel(r"$x^2$")
    ax.set_title(rf"Circular dam break Eq.(27), bed '{tag}', $t={T_END:g}$ s",
                 fontsize=12)
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"pinn_dam_circ_{tag}_field.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    for i, tv in enumerate(ev["t_list"]):
        ax.plot(ev["r_prof"][i], ev["h_prof"][i], lw=1.7, label=rf"$t={tv:g}$ s")
    ax.axvline(R0, color="k", ls="--", lw=1.0, label=rf"$R_0={R0:g}$ m")
    ax.set_xlabel(r"$r$  [m]")
    ax.set_ylabel(r"radially averaged $h$  [m]")
    ax.set_title(rf"Radial profiles, bed '{tag}'")
    ax.grid(True, ls=":", alpha=0.6)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"pinn_dam_circ_{tag}_radial.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    it = np.array(hist["it"])
    for key, lab, style in (("total", r"Total $L$", "-"),
                            ("pde", r"$L_{\mathrm{pde}}$", "--"),
                            ("ic", r"$L_{\mathrm{ic}}$", "-."),
                            ("bc", r"$L_{\mathrm{bc}}$", ":"),
                            ("pos", r"$L_{\mathrm{pos}}$", "-"),
                            ("sa", r"$S(a)$", "-")):
        arr = np.asarray(hist.get(key, []), dtype=float)
        if arr.size == it.size:
            ax.semilogy(it, np.maximum(arr, 1e-16), style, lw=1.4, label=lab)
    ax.axvline(N_ITER, color="gray", lw=0.8, ls="--")
    ax.set_xlabel("iteration")
    ax.set_ylabel("loss (log scale)")
    ax.set_title(rf"Convergence, circular dam break, bed '{tag}'")
    ax.grid(True, which="both", ls=":", alpha=0.6)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"pinn_dam_circ_{tag}_convergence.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.2, 4.2))
    a1.plot(ev["t_list"], (ev["mass"] - ev["mass"][0]) / ev["mass"][0],
            "o-", color="tab:red", lw=1.8, ms=6)
    a1.axhline(0.0, color="k", lw=0.8, ls="--")
    a1.set_xlabel(r"$t$  [s]")
    a1.set_ylabel(r"$(M(t)-M(0))/M(0)$")
    a1.set_title(r"Mass conservation (exact: 0)", fontsize=11)
    a1.grid(True, ls=":", alpha=0.6)

    a2.plot(ev["t_list"], ev["shock_r"], "s-", color="tab:blue", lw=1.8, ms=5)
    a2.axhline(R0, color="k", ls="--", lw=1.0)
    a2.text(ev["t_list"][0], R0, r"  $R_0$", fontsize=9, va="bottom")
    a2.set_xlabel(r"$t$  [s]")
    a2.set_ylabel(r"estimated shock radius  [m]")
    a2.set_title("Shock position", fontsize=11)
    a2.grid(True, ls=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"pinn_dam_circ_{tag}_mass_shock.png"),
                dpi=150, bbox_inches="tight")
    plt.close()


# ============================================================================
# 11b. 激波传播快照与锋位演化图                                    # <<< SHOCK_SNAP v1 >>>
# ============================================================================
def _pad_ragged(seq):
    """把长度不一的序列补成 (n, Lmax) 的 float 数组，短行补 NaN。

    径向平均剖面在域内按固定 bin 统计，理论上等长；补成矩形只是为了让
    npz 免 pickle 保存（np.savez 对 object 数组需要 allow_pickle）。
    """
    arrs = [np.asarray(a, dtype=float).ravel() for a in seq]
    lmax = max([a.size for a in arrs] + [0])
    out = np.full((len(arrs), lmax), np.nan, dtype=float)
    for i, a in enumerate(arrs):
        out[i, :a.size] = a
    return out


def make_shock_plots(ev, outdir, tag_extra=""):
    r"""激波传播图（默认 t = 0.1, 0.2, 0.3, 0.4, 0.5 s）。

    图 1  ..._shock_snapshots.png  (2 x nt)
      上排：h(x,y) 二维场；白色虚线 = 初始坝圈 R0；红色虚线 = 检测到的激波锋
      下排：径向平均剖面 h(r)；灰色点线 = R0；红点 = 激波锋位置
    图 2  ..._shock_evolution.png  (1 x 3)
      (a) 各时刻径向剖面叠加     (b) 过圆心的切面（激波更锐，不被径向平均抹平）
      (c) 激波半径 r_shock(t) 与线性拟合给出的平均锋速

    注：全部量取自 evaluate_field，不引入任何新的物理假设。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 11})
    xs = np.asarray(ev["x"], dtype=float)
    ys = np.asarray(ev["y"], dtype=float)
    ts = np.asarray(ev["t_list"], dtype=float)
    hh = np.asarray(ev["h"], dtype=float)
    nt = len(ts)
    tag = BED
    cols = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
    circ = np.linspace(0.0, 2.0 * np.pi, 400)

    h_lo = float(np.min(hh))
    h_hi = float(np.max(hh))
    if h_hi - h_lo < 1e-12:
        h_hi = h_lo + 1.0
    pad = 0.05 * (h_hi - h_lo)

    # 用 constrained_layout：# <<< SHOCK_LAYOUT v1 >>>
    # colorbar 由 axes 列表生成时会用嵌套 gridspec，与 tight_layout 不兼容。
    fig, axes = plt.subplots(2, nt, figsize=(2.9 * nt, 6.8),
                             squeeze=False, constrained_layout=True)
    im = None
    for k in range(nt):
        a0 = axes[0][k]
        im = a0.pcolormesh(xs, ys, hh[k].T, cmap="viridis", shading="auto",
                           vmin=h_lo, vmax=h_hi)
        a0.plot(XC + R0 * np.cos(circ), YC + R0 * np.sin(circ), "w--", lw=1.1)
        rs = float(ev["shock_r"][k])
        if np.isfinite(rs):
            a0.plot(XC + rs * np.cos(circ), YC + rs * np.sin(circ),
                    color="tab:red", ls="--", lw=1.1)
        a0.set_title(r"$t = %g$ s" % ts[k], fontsize=11)
        a0.set_aspect("equal")
        a0.tick_params(labelsize=9)
        if k == 0:
            a0.set_ylabel(r"$x^2$", fontsize=10)
        else:
            a0.set_yticklabels([])

        a1 = axes[1][k]
        rp = np.asarray(ev["r_prof"][k], dtype=float)
        hp = np.asarray(ev["h_prof"][k], dtype=float)
        a1.plot(rp, hp, color=cols[k % 10], lw=1.6)
        a1.axvline(R0, color="gray", ls=":", lw=1.0)
        if np.isfinite(rs):
            a1.plot([rs], [float(np.interp(rs, rp, hp))], "o",
                    color="tab:red", ms=4)
        a1.set_xlabel(r"$r$  [m]", fontsize=10)
        a1.grid(True, ls=":", alpha=0.6)
        a1.tick_params(labelsize=9)
        a1.set_ylim(h_lo - pad, h_hi + pad)
        if k == 0:
            a1.set_ylabel(r"radially averaged $h$  [m]", fontsize=10)
        else:
            a1.set_yticklabels([])

    fig.colorbar(im, ax=axes[0].tolist(), shrink=0.85, pad=0.015,
                 label=r"$h_\theta$  [m]")
    fig.suptitle(r"Circular dam break Eq.(27), bed '%s': depth snapshots "
                 r"(white dashed: initial dam radius; red dashed: "
                 r"detected front)%s"
                 % (tag, tag_extra), fontsize=12)
    plt.savefig(os.path.join(outdir,
                             "pinn_dam_circ_%s_shock_snapshots.png" % tag),
                dpi=150, bbox_inches="tight")
    plt.close()

    fig, (a0, a1, a2) = plt.subplots(1, 3, figsize=(13.8, 4.3))
    for k in range(nt):
        a0.plot(np.asarray(ev["r_prof"][k], dtype=float),
                np.asarray(ev["h_prof"][k], dtype=float),
                color=cols[k % 10], lw=1.8, label=r"$t = %g$ s" % ts[k])
    a0.axvline(R0, color="k", ls="--", lw=1.0, label=r"$R_0 = %g$ m" % R0)
    a0.set_xlabel(r"$r$  [m]")
    a0.set_ylabel(r"radially averaged $h$  [m]")
    a0.set_title("Radial profiles", fontsize=11)
    a0.grid(True, ls=":", alpha=0.6)
    a0.legend(fontsize=9)

    iy = int(np.argmin(np.abs(ys - YC)))
    for k in range(nt):
        a1.plot(xs - XC, hh[k][:, iy], color=cols[k % 10], lw=1.7,
                label=r"$t = %g$ s" % ts[k])
    for sgn in (-1.0, 1.0):
        a1.axvline(sgn * R0, color="k", ls="--", lw=1.0)
    a1.set_xlabel(r"$x^1 - x^1_c$  [m]")
    a1.set_ylabel(r"$h$ along $x^2 = x^2_c$  [m]")
    a1.set_title("Cross-section through the centre", fontsize=11)
    a1.grid(True, ls=":", alpha=0.6)
    a1.legend(fontsize=9)

    rr = np.asarray(ev["shock_r"], dtype=float)
    a2.plot(ts, rr, "o-", color="tab:red", lw=1.8, ms=6, label="PINN front")
    a2.axhline(R0, color="k", ls="--", lw=1.0)
    ok = np.isfinite(rr)
    if int(ok.sum()) >= 2:
        cf = np.polyfit(ts[ok], rr[ok], 1)
        a2.plot(ts[ok], np.polyval(cf, ts[ok]), color="tab:blue", ls=":",
                lw=1.6, label="linear fit: %.3f m/s" % cf[0])
    a2.set_xlabel(r"$t$  [s]")
    a2.set_ylabel(r"shock radius  [m]")
    a2.set_title("Front propagation", fontsize=11)
    a2.grid(True, ls=":", alpha=0.6)
    a2.legend(fontsize=9)

    fig.suptitle(r"Shock propagation, bed '%s', $t \in \{%s\}$ s%s"
                 % (tag, ", ".join("%g" % t for t in ts), tag_extra), fontsize=12)
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    plt.savefig(os.path.join(outdir,
                             "pinn_dam_circ_%s_shock_evolution.png" % tag),
                dpi=150, bbox_inches="tight")
    plt.close()


def make_shock_single_plots(ev, outdir, tag_extra=""):
    r"""每个时刻**单独**两张图，便于在论文里逐张排版（默认 t = 0.1..0.5 s）。

      pinn_dam_circ_<bed>_shock_t<t>_field.png    平面水深场（方形）
      pinn_dam_circ_<bed>_shock_t<t>_profile.png  径向平均剖面

    文件名里的点写成 'p'（0.1 -> t0p1），避免与扩展名混淆。
    数据全部取自 evaluate_field，与联排图 make_shock_plots 同源，量值一致。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 11})
    xs = np.asarray(ev["x"], dtype=float)
    ys = np.asarray(ev["y"], dtype=float)
    ts = np.asarray(ev["t_list"], dtype=float)
    hh = np.asarray(ev["h"], dtype=float)
    nt = len(ts)
    tag = BED
    circ = np.linspace(0.0, 2.0 * np.pi, 400)

    h_lo = float(np.min(hh))
    h_hi = float(np.max(hh))
    if h_hi - h_lo < 1e-12:
        h_hi = h_lo + 1.0
    pad = 0.05 * (h_hi - h_lo)

    written = []
    for k in range(nt):
        tstr = ("%g" % ts[k]).replace("-", "m").replace(".", "p")
        base = "pinn_dam_circ_%s_shock_t%s" % (tag, tstr)

        # ---- (a) 平面水深场 ----
        fig, ax = plt.subplots(figsize=(4.7, 4.3), constrained_layout=True)
        im = ax.pcolormesh(xs, ys, hh[k].T, cmap="viridis", shading="auto",
                           vmin=h_lo, vmax=h_hi)
        ax.plot(XC + R0 * np.cos(circ), YC + R0 * np.sin(circ), "w--", lw=1.2)
        rs = float(ev["shock_r"][k])
        if np.isfinite(rs):
            ax.plot(XC + rs * np.cos(circ), YC + rs * np.sin(circ),
                    color="tab:red", ls="--", lw=1.2)
        ax.set_aspect("equal")
        ax.set_xlabel(r"$x^1$  [m]", fontsize=10)
        ax.set_ylabel(r"$x^2$  [m]", fontsize=10)
        ax.set_title(r"$t = %g$ s%s" % (ts[k], tag_extra), fontsize=11)
        ax.tick_params(labelsize=9)
        fig.colorbar(im, ax=ax, shrink=0.9, pad=0.02, label=r"$h_\theta$  [m]")
        p = os.path.join(outdir, base + "_field.png")
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        written.append(p)

        # ---- (b) 径向平均剖面 ----
        fig, ax = plt.subplots(figsize=(4.7, 3.6), constrained_layout=True)
        rp = np.asarray(ev["r_prof"][k], dtype=float)
        hp = np.asarray(ev["h_prof"][k], dtype=float)
        ax.plot(rp, hp, color="tab:blue", lw=1.8)
        ax.axvline(R0, color="gray", ls=":", lw=1.0,
                   label=r"$R_0 = %g$ m" % R0)
        if np.isfinite(rs):
            ax.plot([rs], [float(np.interp(rs, rp, hp))], "o",
                    color="tab:red", ms=5, label="detected front")
        ax.set_xlabel(r"$r$  [m]", fontsize=10)
        ax.set_ylabel(r"radially averaged $h$  [m]", fontsize=10)
        ax.set_title(r"$t = %g$ s%s" % (ts[k], tag_extra), fontsize=11)
        ax.grid(True, ls=":", alpha=0.6)
        ax.tick_params(labelsize=9)
        ax.set_ylim(h_lo - pad, h_hi + pad)
        ax.legend(fontsize=9)
        p = os.path.join(outdir, base + "_profile.png")
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        written.append(p)

    return written


# ============================================================================
# 11c. 时域混叠检测与守恒核验                              # <<< REFORM v1 (E1/E2) >>>
# ============================================================================
def _lpde_of(net, X, chunk=8192):
    r"""在给定点集上算 L_pde（分批，避免一次性二阶导爆显存）。

    chunk 必须与 N_INT **解耦**。实测：float64 下 6x128 网络的二阶导反向图
    每点约 75 kB（中间量含度量、Christoffel、梯度乘积），32768 点整批
    约 2.4 GB；而采样审计还要在独立点集上再算一次（10 倍点集直接 23 GB
    → CUDA OOM）。取 8192（约 0.6 GB/批）。分批只改变求和顺序，
    tot/n 的加权均值在数学上与原式完全一致。
    """
    tot, n = 0.0, 0
    for i in range(0, int(X.shape[0]), int(chunk)):
        Xc = X[i:i + int(chunk)]
        R = residuals(Xc, net)
        R1, R2, R3 = R[0], R[1], R[2]
        s = float(torch.mean(W_PDE_M * R1 ** 2 + R2 ** 2 + R3 ** 2).detach())
        tot += s * Xc.shape[0]
        n += Xc.shape[0]
    return tot / max(n, 1)


def sampling_adequacy_report(net, Xp, n_dense=None, tag=""):
    r"""**时域混叠检测**：在一组完全独立的、高密度的低差异点集上重算 L_pde，
    与训练配点上的 L_pde 比较。

        ratio = L_pde(高密度点) / L_pde(训练配点)

    ratio >> 1 说明训练配点上的小残差是"采样点上的巧合"（混叠）：网络在配点
    平面之间的空档里藏了未被采样的大导数。ratio ~ 1 说明训练配点代表了整个
    时空域。判据：ratio <= 10 通过。
    """
    if n_dense is None:
        n_dense = min(10 * int(Xp.shape[0]), 65536)
    n_dense = int(n_dense)
    if n_dense & (n_dense - 1):                  # 非 2 的幂 -> 向下取 2 的幂，
        n_dense = 1 << (n_dense.bit_length() - 1)
    n_dense = max(n_dense, 4096)                 # 使 Sobol 的平衡性质成立
    Xd = _in_box(_lowdisc(n_dense, 3, SEED + 777), three_d=True)
    L_tr = _lpde_of(net, Xp)
    L_de = _lpde_of(net, Xd)
    ratio = L_de / max(L_tr, 1e-300)
    print(f"  [{tag}] L_pde(训练配点 {Xp.shape[0]} 点)        = {L_tr:.6e}")
    print(f"  [{tag}] L_pde(独立高密度 {n_dense} 点) = {L_de:.6e}")
    print(f"  [{tag}] ratio = L_dense / L_train = {ratio:.4f}   "
          + ("通过 ✅（< 10，无混叠迹象）" if ratio < 10.0
             else "失败 ❌（>= 10，疑似时域混叠）"))
    return L_tr, L_de, ratio


def mass_consistency_report(net, n_t=200, ne=161, tag="", criterion=True):
    r"""在 t ~ U(0, T) 的 n_t 个**随机**时刻上核验总质量守恒。

    固壁域上由 Eq.(14) 有 dM/dt = -\oint J q^i n_i ds = 0（q.n = 0），故
    M(t) == M(0) 是精确恒等式。只在少数固定时刻取样时，取样点可能恰好落在
    配点平面上；**随机时刻是非配点取样**，正是暴露时域混叠的量（E2 旧版的
    M(t) 在 t = 0.25/0.5/0.75 s 崩 6%/50%/99%）。判据：相对偏差 < 1e-2。
    """
    xe = np.linspace(X0, X1, int(ne))
    ye = np.linspace(Y0, Y1, int(ne))
    Xg, Yg = np.meshgrid(xe, ye, indexing="ij")
    xy = np.stack([Xg.ravel(), Yg.ravel()], axis=1)
    area = (xe[1] - xe[0]) * (ye[1] - ye[0])
    # 梯形权重（与 evaluate_field 的 M(t) 同口径）        # <<< FIX1 (P2-5a) >>>
    _wx = np.full(xe.size, xe[1] - xe[0])
    _wx[0] *= 0.5
    _wx[-1] *= 0.5
    _wy = np.full(ye.size, ye[1] - ye[0])
    _wy[0] *= 0.5
    _wy[-1] *= 0.5
    _W2t = torch.tensor(np.outer(_wx, _wy).reshape(-1), dtype=torch.float64)

    def _M(t):
        X = torch.tensor(np.column_stack([xy, np.full(xy.shape[0], float(t))]),
                         dtype=torch.float64)
        with torch.no_grad():
            h, _, _ = net(X)
            geo = geometry(X[:, 0], X[:, 1])
            return float(torch.sum(h * geo.J * _W2t).detach())

    rs = np.random.RandomState(SEED + 4242)
    tv = np.sort(rs.uniform(0.0, T_END, int(n_t)))
    M0 = _M(0.0)
    Ms = np.array([_M(float(t)) for t in tv])
    rel = np.abs(Ms - M0) / max(abs(M0), 1e-30)
    worst = float(rel.max())
    kmax = int(rel.argmax())
    print(f"  [{tag}] 随机时刻质量核验（{n_t} 个 t ~ U(0,{T_END:g})，"
          f"评估网格 {ne}x{ne}）：")
    print(f"  [{tag}]   M(0) = {M0:.8f}")
    verdict = (("通过 ✅（< 1e-2）" if worst < 1e-2 else "失败 ❌（>= 1e-2）")
               if criterion else
               "—（训练前基线：初始网络无物理意义，不作判据）")
    print(f"  [{tag}]   max |M(t)-M(0)|/M(0) = {worst:.4e}  "
          f"(最差 t = {tv[kmax]:.4f} s)   " + verdict)
    return M0, tv, Ms, worst


# ============================================================================
# 11d. 3D 水面高度图（对齐 Tian2025 图 4 的 plot_surface）    # <<< SURF3D v1 >>>
# ============================================================================
def _decim(n, target):
    return max(1, int(n) // int(target))


def _surface_cfg():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10})
    return plt


def make_surface_evolution_plot(ev, outdir, tag_extra=""):
    r"""逐时刻 3D 水面高度总览（Tian2025 dCDB9 图 4 的 plot_surface 风格）。

    每个时刻一个 3D 子图：竖轴 = **法向水深 h**（本算例的求解变量），底面 =
    (x^1, x^2)。用于直观看到溃坝波从圆心向外传播的整个过程。
    """
    plt = _surface_cfg()
    xs = np.asarray(ev["x"], dtype=float)
    ys = np.asarray(ev["y"], dtype=float)
    ts = np.asarray(ev["t_list"], dtype=float)
    hh = np.asarray(ev["h"], dtype=float)
    nt = len(ts)
    tag = BED
    sx = _decim(len(xs), 81)
    sy = _decim(len(ys), 81)
    Xg, Yg = np.meshgrid(xs[::sx], ys[::sy], indexing="ij")
    h_lo = float(np.nanmin(hh))
    h_hi = float(np.nanmax(hh))
    if h_hi - h_lo < 1e-12:
        h_hi = h_lo + 1.0
    cmap = plt.get_cmap("Blues")
    norm = plt.Normalize(vmin=h_lo, vmax=h_hi)

    ncol = min(5, nt)
    nrow = int(np.ceil(nt / ncol))
    fig = plt.figure(figsize=(3.3 * ncol, 3.0 * nrow))
    for k in range(nt):
        ax = fig.add_subplot(nrow, ncol, k + 1, projection="3d")
        ax.plot_surface(Xg, Yg, hh[k][::sx, ::sy], cmap=cmap, norm=norm,
                        rstride=1, cstride=1, linewidth=0.0,
                        antialiased=False, shade=False)
        ax.plot(XC + R0 * np.cos(np.linspace(0, 2 * np.pi, 200)),
                YC + R0 * np.sin(np.linspace(0, 2 * np.pi, 200)),
                h_lo, color="0.55", ls="--", lw=0.8, zorder=0)
        ax.set_title(r"$t=%g$ s" % ts[k], fontsize=10, pad=-4)
        ax.set_xlabel(r"$x^1$", fontsize=8, labelpad=-8)
        ax.set_ylabel(r"$x^2$", fontsize=8, labelpad=-8)
        ax.set_zlabel(r"$h$ [m]", fontsize=8, labelpad=-8)
        ax.set_zlim(h_lo, h_hi)
        ax.tick_params(labelsize=7, pad=-2)
        ax.view_init(elev=30, azim=-60)
        try:
            ax.set_box_aspect((1.0, 1.0, 0.55))
        except Exception:
            pass
    fig.subplots_adjust(right=0.90)
    cax = fig.add_axes([0.915, 0.15, 0.013, 0.70])
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                 label=r"$h$ [m]")
    fig.suptitle(r"Circular dam break Eq.(27), bed '%s': 3-D water surface "
                 r"$h(x^1,x^2,t)$%s" % (tag, tag_extra), fontsize=12)
    fn = os.path.join(outdir,
                      "pinn_dam_circ_%s_surface3d_evolution.png" % tag)
    plt.savefig(fn, dpi=150, bbox_inches="tight")
    plt.close()
    return fn


def make_surface_zeta_plots(ev, outdir, tag_extra=""):
    r"""自由水面高程 zeta 的 3D 演化图（**仅当床非平时才画**）。

    零床（flat）上 gamma = 1、z = 0，于是 zeta = z + gamma h == h，与同组的
    h 图完全冗余；此时本函数直接返回空列表并打印说明，不产出重复图件。
    非平床上 zeta 含床面落差，正是 Bachini2020 Fig.4 的口径；竖向放大 _VE 倍
    （与 E2P 的 VE = 2.0 同口径）。
    """
    _VE = 2.0
    plt = _surface_cfg()
    xs = np.asarray(ev["x"], dtype=float)
    ys = np.asarray(ev["y"], dtype=float)
    ts = np.asarray(ev["t_list"], dtype=float)
    zz = np.asarray(ev["zeta"], dtype=float)
    hh = np.asarray(ev["h"], dtype=float)
    amp = float(np.nanmax(np.abs(zz - hh)))
    if amp < 1e-9:
        print("  [3D-zeta] 床为平（max|zeta-h| = %.2e < 1e-9），"
              "zeta 图与 h 图冗余，跳过。" % amp)
        return []
    tag = BED
    nt = len(ts)
    sx = _decim(len(xs), 81)
    sy = _decim(len(ys), 81)
    Xg, Yg = np.meshgrid(xs[::sx], ys[::sy], indexing="ij")
    F = _VE * zz[:, ::sx, ::sy]
    vlo = float(np.nanmin(F))
    vhi = float(np.nanmax(F))
    if vhi - vlo < 1e-12:
        vhi = vlo + 1.0
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(vmin=vlo, vmax=vhi)
    ncol = min(5, nt)
    nrow = int(np.ceil(nt / ncol))
    fig = plt.figure(figsize=(3.3 * ncol, 3.0 * nrow))
    circle = np.linspace(0.0, 2.0 * np.pi, 400)
    for k in range(nt):
        ax = fig.add_subplot(nrow, ncol, k + 1, projection="3d")
        ax.plot_surface(Xg, Yg, F[k], cmap=cmap, norm=norm,
                        rstride=1, cstride=1, linewidth=0.0,
                        antialiased=False, shade=False)
        ax.plot(XC + R0 * np.cos(circle), YC + R0 * np.sin(circle),
                np.full_like(circle, vlo), color="tab:red", ls="--", lw=0.9)
        ax.set_title(r"$t=%g$ s" % ts[k], fontsize=10, pad=-4)
        ax.set_xlabel(r"$x^1$", fontsize=8, labelpad=-8)
        ax.set_ylabel(r"$x^2$", fontsize=8, labelpad=-8)
        ax.set_zlabel("", fontsize=8)
        ax.set_zlim(vlo, vhi)
        ax.tick_params(labelsize=7, pad=-2)
        ax.view_init(elev=30, azim=-60)
        try:
            ax.set_box_aspect((1.0, 1.0, 0.55))
        except Exception:
            pass
    fig.subplots_adjust(right=0.90)
    cax = fig.add_axes([0.915, 0.15, 0.013, 0.70])
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                 label=r"$\zeta\times%g$ [m]" % _VE)
    fig.suptitle(r"Circular dam break, bed '%s': free surface "
                 r"$\zeta=z+\gamma h$ ($\times%g$)%s" % (tag, _VE, tag_extra),
                 fontsize=12)
    fn = os.path.join(outdir,
                      "pinn_dam_circ_%s_surface3d_zeta_evolution.png" % tag)
    plt.savefig(fn, dpi=150, bbox_inches="tight")
    plt.close()
    return [fn]


def make_surface_single_plots(ev, outdir, tag_extra=""):
    r"""每个时刻**单独**一张 3D 水面高度图，便于逐帧查看波的传播。"""
    plt = _surface_cfg()
    xs = np.asarray(ev["x"], dtype=float)
    ys = np.asarray(ev["y"], dtype=float)
    ts = np.asarray(ev["t_list"], dtype=float)
    hh = np.asarray(ev["h"], dtype=float)
    nt = len(ts)
    tag = BED
    sx = _decim(len(xs), 121)
    sy = _decim(len(ys), 121)
    Xg, Yg = np.meshgrid(xs[::sx], ys[::sy], indexing="ij")
    h_lo = float(np.nanmin(hh))
    h_hi = float(np.nanmax(hh))
    if h_hi - h_lo < 1e-12:
        h_hi = h_lo + 1.0
    cmap = plt.get_cmap("Blues")
    norm = plt.Normalize(vmin=h_lo, vmax=h_hi)
    circ = np.linspace(0, 2 * np.pi, 400)

    written = []
    for k in range(nt):
        tstr = ("%g" % ts[k]).replace("-", "m").replace(".", "p")
        fig = plt.figure(figsize=(5.6, 4.8))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(Xg, Yg, hh[k][::sx, ::sy], cmap=cmap, norm=norm,
                        rstride=1, cstride=1, linewidth=0.0,
                        antialiased=False, shade=False)
        rs = float(ev["shock_r"][k])
        if np.isfinite(rs):
            ax.plot(XC + rs * np.cos(circ), YC + rs * np.sin(circ),
                    np.full_like(circ, h_lo), color="tab:red", ls="--", lw=1.2)
        ax.set_xlabel(r"$x^1$", fontsize=9, labelpad=-6)
        ax.set_ylabel(r"$x^2$", fontsize=9, labelpad=-6)
        ax.set_zlabel(r"$h$ [m]", fontsize=9, labelpad=-6)
        ax.set_zlim(h_lo, h_hi)
        ax.tick_params(labelsize=8, pad=-2)
        ax.view_init(elev=30, azim=-60)
        try:
            ax.set_box_aspect((1.0, 1.0, 0.55))
        except Exception:
            pass
        ax.set_title(r"Circular dam break Eq.(27), bed '%s': 3-D water "
                     r"surface at $t=%g$ s%s" % (tag, ts[k], tag_extra),
                     fontsize=10)
        fig.subplots_adjust(right=0.86)
        fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                     shrink=0.7, pad=0.10, label=r"$h$ [m]")
        fn = os.path.join(outdir,
                          "pinn_dam_circ_%s_surface3d_t%s.png" % (tag, tstr))
        plt.savefig(fn, dpi=150, bbox_inches="tight")
        plt.close()
        written.append(fn)
    return written


# ============================================================================
# 12. 主程序
# ============================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="E2 / Eq.(27): 圆域溃坝 ISWE-PINN（N-LAAF）")
    p.add_argument("--bed", choices=sorted(BED_JET), default="flat",
                   help="床：flat(Eq.27 原样) / p1 / p2 / p3")
    p.add_argument("--domain", type=str, default=None,
                   help="域 x0,x1,y0,y1；缺省取该床在论文中的声明域")
    p.add_argument("--r0", type=float, default=None, help="溃坝半径 R0")
    p.add_argument("--center", type=str, default=None, help="圆心 xc,yc")
    p.add_argument("--tend", type=float, default=None, help="结束时刻 T")
    p.add_argument("--bc", choices=["wall", "zgrad", "periodic"], default="wall",
                   help="边界条件：wall = 固壁 q.n = 0（默认）/ zgrad = Neumann / "
                        "periodic = 四面周期（对齐 Tian dCDB9，仅平床）")

    p.add_argument("--nlaaf-n", type=float, default=None, help="N-LAAF 缩放因子 n")
    p.add_argument("--wa", type=float, default=None, help="slope recovery 权重 w_a")
    p.add_argument("--w-ic", type=float, default=None, help="初值权重 omega_I")
    p.add_argument("--w-bc", type=float, default=None, help="边界权重 omega_b")
    p.add_argument("--rebalance", action="store_true",
                   help="启用 Wang2021 式梯度重平衡")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--audit-only", action="store_true", help="只跑审计，不训练")
    p.add_argument("--fast", action="store_true", help="小配点冒烟测试")
    p.add_argument("--no-plot", action="store_true", help="不生成图片")
    p.add_argument("--snap", type=str, default=None,
                   help="激波快照时刻（逗号分隔）；默认 %s" % T_SNAP)
    p.add_argument("--outdir", type=str, default=None, help="输出目录")
    p.add_argument("--device", choices=["cpu", "cuda", "auto"],
                   default="cpu",
                   help="计算设备：cpu(默认，与既往结果一致) / cuda / auto")      # <<< DEVICE v1 >>>
    p.add_argument("--ref", type=str, default=None,
                   help="可选参考解 npz（含 x, y, h），用于 MAE/RMSE")
    p.add_argument("--sampler", choices=["sobol", "grid"], default=None,
                   help="配点采样：sobol = 低差异（默认，消除时域混叠）/ "
                        "grid = 旧张量网格（A/B 对照）")            # <<< REFORM v1 >>>
    p.add_argument("--n-int", type=int, default=None,
                   help="内部配点数（默认 %d）" % N_INT)          # <<< REFORM v1 >>>
    p.add_argument("--n-bc-side", type=int, default=None,
                   help="每边界面配点数（默认 %d）" % N_BC_SIDE)   # <<< REFORM v1 >>>
    return p.parse_args()



def apply_cli(args):
    """把命令行覆盖写回全局量（必须在建网络/配点之前调用）。"""
    global BED, X0, X1, Y0, Y1, XC, YC, R0, T_END, BC_MODE, REBALANCE
    global N_SCALE, W_A, W_I, W_B, N_ITER, NX, NY, NT, M_BC
    global WIDTH, LAYERS, LBFGS_MAXITER, LBFGS_MAXFUN, SEED
    global T_SNAP
    global N_INT, N_IC, N_BC_SIDE, SAMPLER, PERIODIC, W_PDE_M   # <<< REFORM v1 >>>
    global DEVICE                             # <<< DEVICE v1 >>>


    BED = args.bed
    d = BED_DEFAULTS[BED]
    X0, X1, Y0, Y1 = d["domain"]
    R0 = d["r0"]
    if args.domain:
        X0, X1, Y0, Y1 = [float(v) for v in args.domain.split(",")]
    if args.r0 is not None:
        R0 = float(args.r0)
    XC, YC = 0.5 * (X0 + X1), 0.5 * (Y0 + Y1)
    if args.center:
        XC, YC = [float(v) for v in args.center.split(",")]
    if args.tend is not None:
        T_END = float(args.tend)
    if args.nlaaf_n is not None:
        N_SCALE = float(args.nlaaf_n)
    if args.wa is not None:
        W_A = float(args.wa)
    if args.w_ic is not None:
        W_I = float(args.w_ic)
    if args.w_bc is not None:
        W_B = float(args.w_bc)
    BC_MODE = args.bc
    REBALANCE = bool(args.rebalance)
    # ---- <<< REFORM v1 >>> 采样器与周期边界 ----
    if getattr(args, "sampler", None) is not None:
        SAMPLER = str(args.sampler)
    if getattr(args, "n_int", None) is not None:
        N_INT = int(args.n_int)
    if getattr(args, "n_bc_side", None) is not None:
        N_BC_SIDE = int(args.n_bc_side)
    PERIODIC = (BC_MODE == "periodic")
    if PERIODIC and BED != "flat":
        raise SystemExit(
            "  [周期边界] 拒绝：仅平床（z 恒为 0）合法。\n"
            "  曲床上左右 / 上下两条边在诱导度量下**不等距**（z_{s^1} 在两端不同，\n"
            "  例如 p1 上 z_1(0) 与 z_1(10) 不相等），把两侧粘合成环面会使度量在\n"
            "  缝合线上跳跃，周期条件在数学上不成立。请改用 --bc wall 或 --bc zgrad。")

    if args.seed is not None:
        SEED = int(args.seed)
    if args.snap is not None:
        T_SNAP = [float(v) for v in args.snap.split(",") if v.strip()]

    # ---- 设备选择（须在建网络/配点/审计之前生效）  # <<< DEVICE v1 >>>
    _dev = str(args.device).lower()
    if _dev == "auto":
        _dev = "cuda" if torch.cuda.is_available() else "cpu"
    if _dev == "cuda" and not torch.cuda.is_available():
        print("  [警告] --device cuda 但 CUDA 不可用，回退 CPU。")
        _dev = "cpu"
    DEVICE = torch.device(_dev)
    if DEVICE.type == "cuda":
        torch.set_default_device(str(DEVICE))

    if args.fast:
        N_ITER = 200
        NX, NY, NT, M_BC = 16, 16, 5, 16
        N_INT, N_IC, N_BC_SIDE = 4000, 512, 256            # <<< REFORM v1 >>>
        WIDTH, LAYERS = 48, 3
        LBFGS_MAXITER, LBFGS_MAXFUN = 50, 400



def header_report():
    print("=" * 78)
    print("  E2 / Eq.(27): 圆域溃坝（间断解）ISWE-PINN —— N-LAAF 激活")
    print("=" * 78)
    print(f"  床        : {BED}  ({BED_DEFAULTS[BED]['note']})")
    print(f"  域        : [{X0:g}, {X1:g}] x [{Y0:g}, {Y1:g}]"
          f"   (该床的论文声明域: {BED_DEFAULTS[BED]['domain']})")
    print(f"  初始条件  : h = {H_IC:g} 当 (x-xc)^2+(y-yc)^2 <= R0^2，否则 {L_IC:g}；"
          f"u = v = 0     [Eq.(27)]")
    print(f"  圆心/半径 : ({XC:g}, {YC:g}) / R0 = {R0:g}")
    print(f"  时间区间  : [0, {T_END:g}] s")
    print(f"  网络      : {LAYERS} 隐层 x {WIDTH}，N-LAAF (n = {N_SCALE:g})，"
          f"w_a = {W_A:g}       [Eq.(12) + Jagtap2020]")
    print(f"  未知量    : h = 法向水深（网络直接输出；溃坝无静水表面，不用 Eq.(13) 拟设）")
    print(f"  损失      : {W_F:g} L_pde + {W_I:g} L_ic + {W_B:g} L_bc "
          f"+ {W_POS:g} L_pos + {W_A:g} S(a)      [Eq.(16) + Jagtap2020]")
    print(f"  边界      : {BC_MODE}；重平衡: {'开' if REBALANCE else '关'}")
    print(f"  配点采样  : {SAMPLER}"
          + (f"  (Nf = {N_INT}, NI = {N_IC}, Nb = {4 * N_BC_SIDE})"
             if SAMPLER != "grid" else
             f"  (张量网格 {NX}x{NY}x{NT} = {NX * NY * NT})"))
    print(f"  损失权重  : L_pde 内 R1 : R2 : R3 = {W_PDE_M:g} : 1 : 1；"
          f"W_A(S(a)) = {W_A:g}" + ("（已移除 S(a)）" if W_A == 0.0 else ""))

    _dname = torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "CPU"
    print(f"  设备      : {DEVICE.type}  ({_dname})")                 # <<< DEVICE v1 >>>
    print("=" * 78)


def main():
    args = parse_args()
    apply_cli(args)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    outdir = args.outdir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "out")
    os.makedirs(outdir, exist_ok=True)

    header_report()
    run_audits()
    if args.audit_only:
        print("  --audit-only：审计完成，未进入训练。")
        return

    with torch.device("cpu"):          # <<< DEVICE v2 >>>
        net = NLAFNet()                # 权重初始化固定在 CPU 上（见下）
    # 为何固定在 CPU 初始化：cuda 张量用 CUDA RNG、cpu 张量用 CPU RNG，
    # 同一个 torch.manual_seed(SEED) 会给出**不同**的初始化序列。
    # 钉在 CPU 后，--device cuda 与 --device cpu 从同一初始权重出发，
    # 既满足论文的可复现性声明，也让两者的损失轨迹可逐点对照。
    net = net.to(DEVICE)
    Xp, Xi, Xb1, Xb2, Xb = make_points()
    print(f"  配点: Nf = {Xp.shape[0]}, NI = {Xi.shape[0]}, "
          f"Nb = {Xb.shape[0]} (s1 面 {Xb1.shape[0]} + s2 面 {Xb2.shape[0]})")

    # ---- 采样充分性 / 混叠检测（训练前基线）                 # <<< REFORM v1 (E1) >>>
    print()
    print("=" * 78)
    print("  采样充分性审计（时域混叠检测）—— 训练前")
    print("=" * 78)
    sampling_adequacy_report(net, Xp, tag="训练前")
    mass_consistency_report(net, tag="训练前", criterion=False)

    def loss_val(net_):

        return total_loss(net_, Xp, Xi, Xb1, Xb2)

    # ---------------- Adam ----------------
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=max(N_ITER // 2, 1), gamma=1e-1)
    hist = {"it": [], "total": [], "pde": [], "ic": [], "bc": [], "pos": [], "sa": []}

    print()
    print("=" * 78)
    print(f"  Adam: {N_ITER} 步, lr 1e-3, StepLR x0.1 @ {max(N_ITER // 2, 1)}")
    print("=" * 78)
    t0 = time.time()
    for it in range(N_ITER):
        optimizer.zero_grad(set_to_none=True)
        L, Lp, Lic, Lbc, Lpos, La = loss_val(net)
        L.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()
        sched.step()

        if it % RECORD_EVERY == 0:
            hist["it"].append(it)
            for k, v in (("total", L), ("pde", Lp), ("ic", Lic),
                         ("bc", Lbc), ("pos", Lpos), ("sa", La)):
                hist[k].append(float(v.detach()))
        if it % LOG_EVERY == 0:
            print(f"  step {it:6d} | L {float(L.detach()):.4e} | "
                  f"pde {float(Lp.detach()):.4e} | ic {float(Lic.detach()):.4e} | "
                  f"bc {float(Lbc.detach()):.4e} | S(a) {float(La.detach()):.4e} | "
                  f"({time.time() - t0:.0f}s)")
        if REBALANCE and it > 0 and it % RB_EVERY == 0:
            gn = grad_norm_report(net, Xp, Xi, Xb1, Xb2, do_rebalance=True)
            print(f"    [重平衡 @{it}] grad_norms = "
                  + ", ".join(f"{k}={v:.3e}" for k, v in gn.items())
                  + f"  ->  w_I={W_I:.3g}, w_b={W_B:.3g}, w_pos={W_POS:.3g}")

    # ---------------- L-BFGS-B ----------------
    print()
    print("=" * 78)
    print(f"  L-BFGS-B (maxiter={LBFGS_MAXITER}, maxfun={LBFGS_MAXFUN}, "
          f"ftol=gtol=1e-12)")
    print("=" * 78)
    from scipy.optimize import minimize as scipy_minimize

    params = list(net.parameters())
    shapes = [q.shape for q in params]
    sizes = [int(np.prod(s)) for s in shapes]
    offsets = np.cumsum([0] + sizes)

    def flatten_params(ps):
        return torch.cat([q.reshape(-1) for q in ps]).detach().cpu().numpy().astype(np.float64)

    def unflatten_assign(flat):
        ft = torch.tensor(flat, dtype=torch.float64)
        with torch.no_grad():
            for i, q in enumerate(params):
                q.copy_(ft[offsets[i]:offsets[i + 1]].reshape(shapes[i]))

    def value_and_grad(flat_np):
        unflatten_assign(flat_np)
        for q in params:
            if q.grad is not None:
                q.grad = None
        L = loss_val(net)[0]
        L.backward()
        gf = torch.cat([(q.grad if q.grad is not None
                         else torch.zeros_like(q)).reshape(-1)
                        for q in params]).detach().cpu().numpy().astype(np.float64)
        return float(L.detach()), gf

    lbfgs_n = {"n": 0}

    def lbfgs_callback(xk):
        lbfgs_n["n"] += 1
        unflatten_assign(xk)
        L, Lp, Lic, Lbc, Lpos, La = loss_val(net)
        it_lb = N_ITER + lbfgs_n["n"]
        if lbfgs_n["n"] % LOG_EVERY == 0:
            print(f"  iter {it_lb:6d} | L {float(L):.4e} | pde {float(Lp):.4e} | "
                  f"ic {float(Lic):.4e} | bc {float(Lbc):.4e} | "
                  f"S(a) {float(La):.4e}")
        if lbfgs_n["n"] % RECORD_EVERY == 0:
            hist["it"].append(it_lb)
            for k, v in (("total", L), ("pde", Lp), ("ic", Lic),
                         ("bc", Lbc), ("pos", Lpos), ("sa", La)):
                hist[k].append(float(v))

    res = scipy_minimize(value_and_grad, flatten_params(params), method="L-BFGS-B",
                         jac=True, callback=lbfgs_callback,
                         options={"maxiter": LBFGS_MAXITER, "maxfun": LBFGS_MAXFUN,
                                  "ftol": 1e-12, "gtol": 1e-12})
    unflatten_assign(res.x)
    print(f"  L-BFGS-B 完成: success={bool(res.success)}, iterations={int(res.nit)}, "
          f"objective={float(res.fun):.4e}")

    # ---- 采样充分性 / 混叠检测 + 随机时刻守恒核验（训练后）  # <<< REFORM v1 (E1/E2) >>>
    print()
    print("=" * 78)
    print("  采样充分性审计（时域混叠检测）与守恒核验 —— 训练后")
    print("=" * 78)
    sampling_adequacy_report(net, Xp, tag="训练后")
    mass_consistency_report(net, tag="训练后")

    # ---------------- 评估 ----------------
    # 时间序列取 0.1 s 间隔（含 t=0，与快照表同口径），使"守恒残差 / 误差-时间"
    # 曲线的分辨率与图件一致。                              # <<< FIX1 (P2-7) >>>
    ev = evaluate_field(net, t_list=list(np.linspace(0.0, T_END, 11)))

    ref = None
    if args.ref:
        z = np.load(args.ref)
        ref = {"x": z["x"], "y": z["y"], "h": z["h"]}
    report_eval(ev, ref)

    # ---- 激波传播快照（整个 [0, T] 上每 0.1 s）                      # <<< SHOCK_SNAP v2 >>>
    snap_t = snap_times()

    ev_snap = None
    if snap_t:
        ev_snap = evaluate_field(net, t_list=snap_t)
        np.savez(os.path.join(outdir, "pinn_dam_circ_%s_shock.npz" % BED),
                 x=ev_snap["x"], y=ev_snap["y"], t_snap=ev_snap["t_list"],
                 h_snap=ev_snap["h"], zeta_snap=ev_snap["zeta"],
                 u1_snap=ev_snap["u1"], u2_snap=ev_snap["u2"],
                 mass_snap=ev_snap["mass"], shock_r_snap=ev_snap["shock_r"],
                 r_prof_snap=_pad_ragged(ev_snap["r_prof"]),
                 h_prof_snap=_pad_ragged(ev_snap["h_prof"]),
                 cfg=np.array([X0, X1, Y0, Y1, XC, YC, R0, T_END,
                               H_IC, L_IC, N_SCALE, W_A, W_I, W_B]))
        print("  数据: pinn_dam_circ_%s_shock.npz  (快照 t = %s s)"
              % (BED, ", ".join("%g" % t for t in snap_t)))
    else:
        print("  [提示] T_SNAP 内无落在 [0, T] 中的时刻，跳过激波快照。")

    np.savez(os.path.join(outdir, f"pinn_dam_circ_{BED}.npz"),
             x=ev["x"], y=ev["y"], t_list=ev["t_list"],
             h=ev["h"], u1=ev["u1"], u2=ev["u2"], zeta=ev["zeta"],
             mass=ev["mass"], shock_r=ev["shock_r"], r_prof=ev["r_prof"],
             h_prof=ev["h_prof"],
             hist_it=np.array(hist["it"]), hist_total=np.array(hist["total"]),
             hist_pde=np.array(hist["pde"]), hist_ic=np.array(hist["ic"]),
             hist_bc=np.array(hist["bc"]),
             hist_pos=np.array(hist["pos"]),      # <<< FIX1 (P2-1) >>>
             hist_sa=np.array(hist["sa"]),
             cfg=np.array([X0, X1, Y0, Y1, XC, YC, R0, T_END, H_IC, L_IC,
                           N_SCALE, W_A, W_I, W_B, W_F, W_POS]))
    print(f"  数据: pinn_dam_circ_{BED}.npz")

    if not args.no_plot:
        make_plots(ev, outdir, hist)
        if ev_snap is not None:
            make_shock_plots(ev_snap, outdir)
            make_shock_single_plots(ev_snap, outdir)
            make_surface_evolution_plot(ev_snap, outdir)          # <<< SURF3D v1 >>>
            make_surface_zeta_plots(ev_snap, outdir)              # <<< FIX1 (P2-3) >>>
            make_surface_single_plots(ev_snap, outdir)            # <<< SURF3D v1 >>>
        print(f"  图件: pinn_dam_circ_{BED}_"

              f"{{field,radial,convergence,mass_shock}}.png")
        if ev_snap is not None:
            print(f"  图件: pinn_dam_circ_{BED}_"
                  f"{{shock_snapshots,shock_evolution,surface3d_evolution}}.png"
                  f"  (t = {snap_t})")
            print(f"  图件（每时刻单独一张）: "
                  f"pinn_dam_circ_{BED}_shock_t*.png  "
                  f"({2 * len(snap_t)} 张: 每时刻 field + profile)")
            print(f"  图件（每时刻 3D 水面）: "
                  f"pinn_dam_circ_{BED}_surface3d_t*.png  "
                  f"({len(snap_t)} 张)")

    print("=" * 78)
    print("  全部完成。")
    print("=" * 78)


if __name__ == "__main__":
    main()
