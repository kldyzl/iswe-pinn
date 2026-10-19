#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
E2-P / Bachini2020 §4.2 "parabola case"（无摩擦 = 光滑底床那一支）：
抛物床上的平面溃坝，内蕴浅水方程 PINN，N-LAAF 激活
================================================================================
算例规格（Bachini & Putti, ESAIM: M2AN 54 (2020), p.2150 §4.2 "Test case 1:
parabola case"；提取本见 tex/iswe_numerical_experiments.pdf §3）
  域          U = [0, 10] x [0, 1]                                Eq.(22)
  床          z = (1/25)(x^1 - 10)^2
  溃坝位置    x^1 = 2.0 m（平面坝）
  初始水深    上游 0.5 m (x^1 < 2)，下游 0.2 m (x^1 >= 2)，q = 0
  边界        x^1 = 0 固壁；x^1 = 10 自由出流（透射）；x^2 = 0, 1 固壁
  输出时刻    t = 0, 0.5, 1.0, 1.5 s
  流量截面    x^1 = 2.5, 5.0, 7.5 m
  垂直放大    2.0（对应原文 Figure 4 的 "vertical magnification factor of 2.0"）

【先确认的口径：法向 vs 垂向】—— 结论：**法向**
  ① Bachini2020 p.2148 水柱高度项写作 eta * d x^3 / d s^3，其中 eta 为沿床面法向
     量得的水深、d x^3 / d s^3 = gamma 为单位法向的竖直分量，故**垂向厚度 = gamma*eta**；
  ② p.2149 前言："water depth in any case small enough to exclude the issue of the
     intersection of the local normals so that the coordinate transformation is always
     a diffeomorphism"（水深沿法向量取，否则不存在"局部法线相交"问题）；
  ③ p.2150 §4.2 原文："water depth of 0.5 m upstream and 0.2 m downstream the dam"；
  ④ 本论文 §3.3 "measured along the bed normal"，§4.5 "the vertical thickness of a
     water column is gamma h"。
  => 0.5 / 0.2 按**法向水深 h** 写入初值；垂向厚度为 gamma h（本床上 gamma in [0.7809, 1]）。

论文公式对照（manuscript.tex）
  Eq.(2)   正交 LCS 标架 t_1, t_2, t_3（Gram-Schmidt）
  Eq.(3)   诱导度量 G_sw = diag(h_(1)^2, h_(2)^2)；逆度量 g^{ij}
  Eq.(4)   J = sqrt(det G_sw) = h_(1) h_(2) = sqrt(1 + |grad z|^2)
  Eq.(5)   gamma = d x^3 / d s^3 = 1/J
  Eq.(7)   度量感知算子
             (grad_G f)^i = g^{ij} d_j f
             (div_G v)    = (1/J) d_i(J v^i)
             (div_G T)^j  = (1/J) d_i(J T^{ij}) + Gam^j_ik T^{ik}
  Eq.(9)   内蕴浅水方程（保守形式）
  Eq.(12)   网络输出 h, u^1, u^2；其中 **h 直接就是法向水深**（方程求的未知量）。
            本算例无静水面，Eq.(13) 的参考水位 H_0 无任何作用，故不引入
            （论文 §3.5 原话；论文 §4.5 的溃坝实验也不含 H_0）。
  Eq.(14)(15) 质量 / 动量残差（hybrid 原始--保守装配）
  Eq.(16)  复合损失
  Eq.(22)  床 p1（本算例）
  Jagtap2020   N-LAAF 激活 + slope recovery 项

与 exp_e2_dam/iswe_pinn_e2_circulardam.py 的关系（可信度来源）
  几何层 geometry()、通量与源项 flux_and_source()、残差**装配行**、损失**装配行**、
  N-LAAF 隐藏层与可训练参数、审计 G1-G6 / A7 / N-LAAF、Adam+LBFGS 循环 —— 逐行相同。
  本算例改写五处：
    ① 全局配置（域 / 床 / 坝位 / 时间 / 输出时刻 / 配点 / 边界模式）；
    ② 初值 h_initial（圆台阶 -> 平面台阶，法向水深 0.5 / 0.2）；
    ③ 边界 boundary_terms（新增 mixed：固壁 + 自由出流）；
    ④ **网络输出头**：删去 Eq.(13) 的参考水位 H_0（本算例无静水面），网络第 1 个
       输出改为**法向水深 h 本身**，net() 返回元数 4 -> 3；residuals 的返回值与
       total_loss / grad_norm_report / boundary_terms / evaluate_field 的解包同步调整；
    ⑤ 审计 O2 改写为与 H_0 无关的恒等式（用任意光滑正水深，zeta = z + gamma h 直接算），
       评估与出图（第 11 节起整体替换）。
  该内核已通过三重审计：sympy 符号级、源码级保真（ast 直读）、numpy 独立有限差分，
  并有服务器实测（静水态 max|R2| = 3.553e-15, max|R3| = 0）。

本算例新增的**无需参考解**的诊断量（连续方程积分 + 边界条件的精确推论）
  M(t)      = \int_U h J dx^1 dx^2                     总质量
  Q_out(t)  = \int_{x^1 = X1} h u^1 J dx^2             出口流量
  dM/dt + Q_out = 0                                    质量-出流平衡（精确恒等式）
  另：床与初值均与 x^2 无关 => 精确解也与 x^2 无关，故 max|d_2 h| 应 ~ 0。

用法
  python iswe_pinn_e2p_parabola.py --audit-only      # 只跑审计（数秒）
  python iswe_pinn_e2p_parabola.py --fast            # 小配点冒烟
  python iswe_pinn_e2p_parabola.py --device cuda     # 正式训练（GPU）
  python iswe_pinn_e2p_parabola.py --device cuda --outdir <home>/exp_e2p/out
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

BED = "p1"              # 床的选择，见 BED_JET / BED_DEFAULTS
X0, X1 = 0.0, 10.0      # 域 U = [X0,X1] x [Y0,Y1] = [0,10] x [0,1]   [Eq.(22)]
Y0, Y1 = 0.0, 1.0
XC, YC = 5.0, 0.5        # 域中心：**仅**用于网络输入的归一化（不再是坝心）
DAM_X = 2.0              # 平面坝位置 x^1 = 2.0 m（Bachini2020 §4.2）
H_UP, H_DN = 0.5, 0.2    # 上游 / 下游初始**法向**水深 [m]（Bachini2020 §4.2）
T_END = 1.5              # 时间区间 [0, 1.5] s
# 本算例**没有** Eq.(13) 的参考水位 H_0。
#   参考水位唯一的物理作用是让静止湖面落进残差损失的零空间（良平衡）；而溃坝初值的
#   自由水面 zeta = z + gamma*h 沿 x^1 从约 4.39 m 单调降到 0.2 m、坝线处还有跳跃，
#   **不是静水面**，故 H_0 在本次训练里毫无作用（论文 §3.5 原话：no still-water level
#   时 "H_0 is a fixed reference level of no dynamical significance ... the ansatz is
#   then inert"）。论文 §4.5 的溃坝实验同样通篇不出现 H_0。
#   因此网络**直接输出法向水深 h**（方程求的未知量就是 h），不做任何参考面重构。

# <<< SHOCK_SNAP v2 >>>
# ---- 输出/快照时刻：**整个 [0, T] 上每 0.1 s 一帧** ----
#   T_SNAP = None 时自动取 0.1 的整数倍（含 0）直到 T_END：T_END = 1.5 s
#   -> 0.0, 0.1, ..., 1.5 共 16 帧。每帧出平面水深场 / 纵向剖面 / 3D 水面，
#   全部落在服务器 outdir 里，便于远程直接查看波的传播。
#   （Bachini2020 §4.2 的正文输出时刻 0/0.5/1/1.5 s 是这 16 帧的子集。）
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
                         # 损失项）。E2/E2P 实测该正则后期独占损失的绝大部分预算
                         # （E2 终态 S(a) 占 99.99%），与物理无关。N-LAAF 本体不变。

N_SCALE = 10.0           # N-LAAF 缩放因子 n（Jagtap2020 函数逼近算例取 10）
H_DW = 5.0e-2            # 干湿一致性罚的水深阈值 = 最小物理水深 H_DN 的 1/4。
                         # h >= H_DW 视为"湿"，罚项为 0；真实解 h in [0.2, 0.5] 因此
                         # 完全落在该项作用域之外，只有非物理的干/负水深才被罚。
                         # 权重 w(h) 在 h < 0 处**封顶为 1**（见 total_loss 的注释）。
LBFGS_MAXITER = 30000
LBFGS_MAXFUN = 100000
SEED = 42

# NX = 61 是**特意**选的：配点 x^1 = 10*k/(NX-1)，k = 12 时恰为 2.0，即坝线正好
# 落在一个配点节点上，使初值台阶在采样层面与定义逐字一致（audit_ic_and_points
# 会对这一条做硬断言，NX 被改动时会立即报错）。
NX, NY, NT, M_BC = 61, 60, 10, 60    # 仅 sampler == "grid"（旧张量网格）时使用
# <<< REFORM v1 >>> 配点采样：消除**时域混叠**
#   旧版内部配点是 61x60x10 的**张量网格**，t 方向只有 10 个离散平面，任意两个
#   相邻平面之间 0 个点。网络可以在空档里塞进一段宽 delta << dt 的陡过渡：
#   配点上 d_t h ~ 0（残差被"满足"），空档里 d_t h ~ h/delta 极大。E2P 上一轮
#   终态 M(0) = 2.936 -> t = 1.5 s 时 0.2895（掉 90.1%），而
#   L_pde = 5.6e-3、L_bc = 1.0e-3 **全部"满足"** —— 与 E2 同源的时域混叠
#   （见 AUDIT_E2_vs_dCDB9_20261003.md 第 3 节的构造反例）。
#   改为 3D 低差异（Sobol, scramble）采样后 t 方向边际非退化。配点在 **CPU**
#   上用 numpy/scipy 生成再搬到 DEVICE，故 cpu / cuda 的配点逐位相同。
N_INT = 32768            # 内部配点数（Sobol 3D：域 U x [0,T]；2^15，
                         # 使 Sobol 的平衡性质成立；与 Tian dCDB9 的 32768
                         # 个内部配点一致。禁用 65536：float64 全批反传峰值
                         # 15 GiB/进程，与 E2 同卡并发即 CUDA OOM）
N_IC = 4096              # 初始面 t = 0 的配点数（Sobol 2D）
N_BC_SIDE = 2048         # 每个边界面的配点数（4 个面 -> Nb = 8192）
SAMPLER = "sobol"        # sobol（默认）/ grid（旧张量网格，A/B 对照）
W_PDE_M = 10.0           # L_pde 中质量方程 R1 的权重（对齐 Tian dCDB9 的 10:1）
WIDTH, LAYERS = 128, 6

BC_MODE = "mixed"        # mixed（本算例默认）| wall | zgrad
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
    "p1":   dict(domain=(0.0, 10.0, 0.0, 1.0), dam_x=2.0,
                 note="Bachini2020 §4.2 parabola case（本算例；Eq.(22)）"),
    "flat": dict(domain=(-10.0, 10.0, -10.0, 10.0), dam_x=0.0,
                 note="零床退化检验（经典 SWE，Q1 门槛）"),
    "p2":   dict(domain=(-3.0, 3.0, -3.0, 3.0), dam_x=0.0,
                 note="HCB 中心凸床（Bachini §4.3；仅保留完整性）"),
    "p3":   dict(domain=(-10.0, 10.0, -4.0, 4.0), dam_x=-8.5,
                 note="全三维床（Bachini §4.4，坝在 x^1 = -8.5；仅保留完整性）"),
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
# 3. 初值（Bachini2020 §4.2 parabola case：x^1 = 2 m 处的平面坝）
# ============================================================================
def h_initial(x, y):
    r"""h = H_UP 当 x^1 < DAM_X（上游），否则 H_DN（下游）。

    返回的是**法向**水深 h，与论文 §3.3 / §4.5 的口径一致：自由水面高程为
        zeta = z + gamma h ,
    故本初值**不是**静水态（zeta 沿 x^1 从约 4.39 m 单调降到 0.2 m，坝线两侧还有
    gamma(DAM_X)*(H_UP - H_DN) 的跳跃）。水流自溃坝起即受重力沿坡驱动 —— 这正是
    Bachini2020 §4.2 的设计意图（"gravity-driven fluid in a dam-break setting"）。
    """
    return torch.where(x < DAM_X,
                       torch.full_like(x, H_UP), torch.full_like(x, H_DN))


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

    输出 (h, u^1, u^2)：第 1 个输出**直接就是法向水深 h**（即方程求的未知量）。

        h = 网络的第 1 个输出

    本算例无静水面，论文 Eq.(13) 的参考水位 H_0 在此没有任何作用，故不引入
    （论文 §3.5："no still-water level ... H_0 is a fixed reference level of no
    dynamical significance"；论文 §4.5 的溃坝实验同样不含 H_0）。
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
    mixed : **本算例默认**，逐面指定（对应 Bachini2020 §3.4 的两类边界）
              x^1 = X0      : 固壁      q^1 = 0
              x^2 = Y0, Y1  : 固壁      q^2 = 0
              x^1 = X1      : 自由出流（透射）—— 沿面内共法向零梯度，
                              d_1(h) = d_1(u^1) = d_1(u^2) = 0，即有限体积中
                              ghost cell 的零阶外插（原文 §3.4: eta_out = eta_in,
                              q_out = q_in）在 PINN 中的对应物。
    三组罚项各自取 mean 后等权平均，避免面元点数不等造成的隐性加权。
    """
    if BC_MODE == "mixed":
        # 面 s^1 = X0（上游端，固壁）与面 s^1 = X1（下游端，自由出流）
        m_lo = (Xb1[:, 0] - X0).abs() < 1e-12
        m_hi = (Xb1[:, 0] - X1).abs() < 1e-12
        hA, u1A, _ = net(Xb1[m_lo])
        L_wall_lo = torch.mean((hA * u1A) ** 2)        # 固壁：q^1 = 0
        hB, _, u2B = net(Xb2)
        L_wall_y = torch.mean((hB * u2B) ** 2)         # 上下壁：q^2 = 0
        Xg = Xb1[m_hi].clone().requires_grad_(True)
        hC, u1C, u2C = net(Xg)
        tot = None
        for f in (hC, u1C, u2C):
            d = torch.autograd.grad(f.sum(), Xg, create_graph=True)[0][:, 0]
            t = torch.mean(d ** 2)
            tot = t if tot is None else tot + t
        L_out = tot / 3.0                              # 自由出流：d_1(.) = 0
        return (L_wall_lo + L_wall_y + L_out) / 3.0

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

    raise ValueError("BC_MODE 必须是 'mixed' / 'wall' / 'zgrad'")


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
    #   Tian 以 10:1 加权，此处照做。
    L_pde = torch.mean(W_PDE_M * R1 ** 2 + R2 ** 2 + R3 ** 2)


    # ---- 初值 @ t = 0：抛物床上 x^1 = 2 处的台阶（法向水深 0.5 / 0.2）----
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
    #   0 <= h < H_DW   : w 由 1 线性降到 0（原实现是它的**平方**，这里保留平方）；
    #   h < 0（非物理） : w = 1（**封顶**）—— h < 0 的不物理程度已由非负性罚项
    #                     relu(-h)^2 单独度量，干湿项不重复计量。若不封顶，以
    #                     H_DW = 0.05 归一化的相对权重在 h = -1 处会达到
    #                     (1.05/0.05)^2 = 441 倍，把"轻度一致性罚"变成主导损失的量。
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
    r"""CPU 上的 d 维低差异序列（Sobol, scramble），返回 (n, d) float64 张量。

    **在 CPU 上生成再搬到 DEVICE** —— CUDA 与 CPU 的 RNG 实现不同，若直接在
    DEVICE 上调 torch.rand，cpu / cuda 会得到不同的配点。用 numpy/scipy 生成
    则两者逐位相同。
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
    r"""快照/输出时刻表。

    T_SNAP 为 None（默认）时自动取 step 的整数倍（**含 0**）直到 T_END
    （T_END = 1.5 -> [0.0, 0.1, ..., 1.5]，共 16 帧）；否则沿用 T_SNAP。
    """
    if T_SNAP is not None:
        return [float(t) for t in T_SNAP if -1e-12 <= t <= T_END + 1e-12]
    n = int(np.floor(T_END / step + 1e-9))
    return [round(k * step, 10) for k in range(0, n + 1)]


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
        Xb1 : 2*N_BC_SIDE 个点（面 s^1 = X0 与 s^1 = X1；前半 / 后半按此顺序）
        Xb2 : 2*N_BC_SIDE 个点（面 s^2 = Y0 与 s^2 = Y1）
        Xb  : Xb1 与 Xb2 的拼接（供统一审计）

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

    # 前半 = 面 x^1 = X0，后半 = 面 x^1 = X1（boundary_terms 的 mixed 分支
    # 用坐标掩码区分，故此顺序必须保持）
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
          f"坝线 x^1 = {DAM_X:g}")
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
    #   故**仅当 z_{s^1} z_{s^2} == 0** 时两套口径逐点相同。本算例的 p1 床
    #   （z = (x^1-10)^2/25，与 x^2 无关）恰好满足 z_2 == 0，故代码口径与论文
    #   逐点一致；换成 z_{s^1} z_{s^2} != 0 的床会引入 O(c) 偏差，这里写成
    #   可执行检查以防静默出错。
    cross_zz = _mx(geo.Bx * geo.By)
    print(f"  [审计:{tag}] (F1) max|z_1 z_2|（坐标口径重合判据）    = {cross_zz:.3e}")

    tol = 1e-9
    ok = (e1 < 1e-12 and e2 < 1e-12 and e3 < 1e-12
          and e4 < tol and e5 < tol and e6 < tol and e7 < tol and e8 < tol)
    # <<< REFORM v1 (F1) >>> p1 床必须满足 z_{s^1} z_{s^2} == 0
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
    r"""核验初值（抛物床平面溃坝）与配点集：

      (a) h 取值恰为 {H_DN, H_UP}；
      (b) **坝线 x^1 = DAM_X 恰好落在训练网格节点上**（NX 的选取保证），
          使初值台阶在采样层面与定义逐字一致；
      (c) 上游面积占比 ≈ (DAM_X - X0)/(X1 - X0)；
      (d) 初始自由水面 zeta = z + gamma h 在坝线两侧的跳跃
          = gamma(DAM_X) * (H_UP - H_DN)（z 连续、h 跳变），与闭式一致；
      (e) 初始总质量 M(0) = \int h J dx^1 dx^2 与**闭式积分**对照：

              \int sqrt(1 + z_1^2) dx^1 = (25/2) * (1/2) *
                    [ u sqrt(1 + u^2) + asinh u ],      u = 2 (x^1 - 10)/25 ,
              M(0) = (Y1 - Y0) * [ H_UP (I(DAM_X) - I(X0))
                                 + H_DN (I(X1) - I(DAM_X)) ] .

          离散一方**在坝线处把区间切成两段**再做复合梯形积分（没有任何单元跨越
          间断），被积函数在每段内光滑，故误差是 O(dx^2)；判据取 1e-6 的相对界，
          比 O(dx) 的界锐利得多，能真正抓住 J、水深幅值或坝位写错这类错误；
      (f) 全部配点在域内；边界点恰好落在四个面上。
    """
    n = 601
    xv = torch.linspace(X0, X1, n, dtype=torch.float64)
    yv = torch.linspace(Y0, Y1, n, dtype=torch.float64)
    Xg, Yg = torch.meshgrid(xv, yv, indexing="ij")
    hv = h_initial(Xg.reshape(-1), Yg.reshape(-1))
    frac_up = float((hv > 0.5 * (H_UP + H_DN)).double().mean())
    frac_ref = float((DAM_X - X0) / (X1 - X0))
    hi, lo = float(hv.max()), float(hv.min())

    # (b) 训练网格上坝线是否为节点（**仅张量网格采样时**才有意义）
    if SAMPLER == "grid":
        xtr = torch.linspace(X0, X1, NX, dtype=torch.float64)
        dam_on_node = bool(((xtr - DAM_X).abs() < 1e-12).any())
    else:
        dam_on_node = True     # <<< REFORM v1 >>> 低差异采样下不作判据


    # (d) 坝线处 zeta 跳跃
    xd = torch.tensor([DAM_X], dtype=torch.float64)
    z_d, z1_d, _, _, _, _ = BED_JET[BED](xd, torch.zeros_like(xd))
    gamma_d = 1.0 / float(torch.sqrt(1.0 + z1_d ** 2))
    zeta_d = float(z_d) + gamma_d * np.array([H_UP, H_DN])
    jump = float(zeta_d[0] - zeta_d[1])
    jump_ref = gamma_d * (H_UP - H_DN)

    # (e) 初始质量的闭式（精确值）
    def _I(x):
        u = 2.0 * (x - 10.0) / 25.0
        return 6.25 * (u * np.sqrt(1.0 + u ** 2) + np.arcsinh(u))

    M_exact = (Y1 - Y0) * (H_UP * (_I(DAM_X) - _I(X0))
                           + H_DN * (_I(X1) - _I(DAM_X)))

    Xp, Xi, Xb1, Xb2, Xb = make_points()
    on_s1 = int((((Xb1[:, 0] - X0).abs() < 1e-12)
                 | ((Xb1[:, 0] - X1).abs() < 1e-12)).sum())
    on_s2 = int((((Xb2[:, 1] - Y0).abs() < 1e-12)
                 | ((Xb2[:, 1] - Y1).abs() < 1e-12)).sum())

    # 离散质量：**在坝线处把区间切成两段**，每段复合梯形积分 —— 没有任何单元跨越
    # 间断，被积函数在段内光滑，故误差是 O(dx^2)（不是 O(dx)）。J 由 BED_JET 的闭式
    # z_1 给出，与 geometry() 是相互独立的实现，因此该式同时核验了床、J 与初值幅值。
    def _trap(y, x):
        y = np.asarray(y, dtype=float)
        x = np.asarray(x, dtype=float)
        return float(np.sum(0.5 * (y[1:] + y[:-1]) * (x[1:] - x[:-1])))

    def _J_of(xv):
        xt = torch.tensor(np.asarray(xv, dtype=np.float64), dtype=torch.float64)
        _, z1v, _, _, _, _ = BED_JET[BED](xt, torch.zeros_like(xt))
        return np.sqrt(1.0 + z1v.detach().cpu().numpy() ** 2)

    mseg = 1000
    x1v = np.linspace(X0, DAM_X, mseg + 1)
    x2v = np.linspace(DAM_X, X1, mseg + 1)
    M_disc = (Y1 - Y0) * (H_UP * _trap(_J_of(x1v), x1v)
                          + H_DN * _trap(_J_of(x2v), x2v))
    rel_M = abs(M_disc - M_exact) / M_exact
    bound_M = 1.0e-6

    # 参考信息：evaluate_field 的求积口径在 FIX1 (P2-5) 之后已由**等权矩形和**
    # （左端点，误差 O(dx)）改成**梯形**；这里把两种口径都算出来并打印，用于解释
    # 评估报告里的质量漂移，并与评估口径对账。        # <<< FIX1b (P2-8) >>>
    _, z1f, _, _, _, _ = BED_JET[BED](Xg.reshape(-1), Yg.reshape(-1))
    Jf = torch.sqrt(1.0 + z1f ** 2)
    dxg = (X1 - X0) / (n - 1)
    dyg = (Y1 - Y0) / (n - 1)
    M_naive = float((hv * Jf).sum()) * dxg * dyg          # 旧口径（等权矩形和）
    _wxg = np.full(n, dxg)
    _wxg[0] *= 0.5
    _wxg[-1] *= 0.5
    _wyg = np.full(n, dyg)
    _wyg[0] *= 0.5
    _wyg[-1] *= 0.5
    # Xg/Yg 由 torch.meshgrid(indexing="ij") 生成 -> 一维索引 k = i*n + j，
    # 与 np.outer(wx, wy).reshape(-1) 的 row-major 顺序一致。
    _Wg = np.outer(_wxg, _wyg).reshape(-1)
    M_eval = float((hv * Jf * torch.tensor(_Wg, dtype=torch.float64)).sum())

    print(f"  [审计:IC] h 取值范围 = [{lo:.6f}, {hi:.6f}]（应恰为 "
          f"[{H_DN:g}, {H_UP:g}]）")
    print(f"  [审计:IC] 坝线 x^1 = {DAM_X:g} 是否为训练网格节点: "
          + (f"{'是' if dam_on_node else '否'}（NX = {NX}）"
             if SAMPLER == "grid" else
             "不适用（sampler = sobol，非张量网格）"))       # <<< REFORM v1 >>>

    print(f"  [审计:IC] 上游面积占比 = {frac_up:.6f}，闭式 {frac_ref:.6f}，"
          f"差 = {abs(frac_up - frac_ref):.3e}（网格离散 O(1/n)）")
    print(f"  [审计:IC] 坝线处 zeta 跳跃 = {jump:.12f}，闭式 gamma*(H_UP-H_DN) = "
          f"{jump_ref:.12f}，差 = {abs(jump - jump_ref):.3e}")
    print(f"  [审计:IC] 初始质量 M(0): 分段梯形 {M_disc:.10f} / 闭式 "
          f"{M_exact:.10f}（相对差 {rel_M:.3e} < {bound_M:.0e}）")
    print(f"  [审计:IC] 参考：评估口径（梯形）{M_eval:.8f}，相对差 "
          f"{abs(M_eval - M_exact) / M_exact:.3e}；旧口径（等权矩形和）"
          f"{M_naive:.8f}，相对差 {abs(M_naive - M_exact) / M_exact:.3e}"
          f"（非判据）")
    print(f"  [审计:IC] 边界点落在面上: s1 面 {on_s1}/{Xb1.shape[0]}, "
          f"s2 面 {on_s2}/{Xb2.shape[0]}")
    print(f"  [审计:IC] 配点规模: Nf={Xp.shape[0]} NI={Xi.shape[0]} "
          f"Nb1={Xb1.shape[0]} Nb2={Xb2.shape[0]} 合计 {Xb.shape[0]}")

    ok = (abs(hi - H_UP) < 1e-12 and abs(lo - H_DN) < 1e-12
          and dam_on_node
          and abs(frac_up - frac_ref) < 5.0 / n
          and abs(jump - jump_ref) < 1e-10
          and rel_M < bound_M
          and on_s1 == Xb1.shape[0] and on_s2 == Xb2.shape[0])
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
# 11. 评估与出图（Bachini2020 §4.2 parabola case：抛物床平面溃坝）
# ============================================================================
XSECT = (2.5, 5.0, 7.5)   # 论文指定的三个流量监测截面 x^1 = 2.5 / 5.0 / 7.5 m
VE = 2.0                  # 自由水面 / 床面的垂直放大因子（Bachini2020 Fig.4: 2.0）


def _trapz(y, x):
    r"""梯形积分（不依赖 numpy 版本：np.trapz 与 np.trapezoid 命名有差异）。"""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * (x[1:] - x[:-1])))


def _tslug(t):
    r"""时刻 -> 文件名片段：0 -> '0'，0.5 -> '0p5'，1.5 -> '1p5'。"""
    return ("%g" % t).replace("-", "m").replace(".", "p")


def profile_x(h2d, y=None):
    r"""截面平均剖面 \bar h(x^1)：本算例的床与初值都与 x^2 无关，故对 x^2 平均。

    y 给定时用**梯形权重**（与 M(t)、Q_out 同口径，消除求积不相容带来的
    O(dy) 伪残差）；y 为 None 时退化为等权平均。        # <<< FIX1 (P2-6) >>>
    """
    a = np.asarray(h2d, dtype=float)
    if y is None:
        return a.mean(axis=1)
    y = np.asarray(y, dtype=float)
    w = np.full(y.size, y[-1] - y[0]) / max(y.size - 1, 1)
    w[0] *= 0.5
    w[-1] *= 0.5
    return (a * w).sum(axis=1) / w.sum()


def bed_profile(xe):
    r"""床面高程 z(x^1) 与 gamma(x^1) = 1/sqrt(1 + z_1^2)（由 BED_JET 闭式给出）。"""
    xt = torch.tensor(np.asarray(xe, dtype=np.float64), dtype=torch.float64)
    z, z1, _, _, _, _ = BED_JET[BED](xt, torch.zeros_like(xt))
    z = z.detach().cpu().numpy()
    gam = 1.0 / np.sqrt(1.0 + z1.detach().cpu().numpy() ** 2)
    return z, gam


def q_section(h_col, u1_col, J_col, ye):
    r"""截面流量 Q = \int_0^1 h u^1 J dx^2   [m^3/s]。

    内蕴推导：面 s^1 = const 上的**单位共法向**（面内、指向域外）为 nu = t_1 / h_(1)，
    弧长元 ds = h_(2) dx^2；由 q = q^i t_i 与正交性 t_1.t_1 = h_(1)^2、t_1.t_2 = 0，
        q . nu = q^i t_i . t_1 / h_(1) = q^1 h_(1)^2 / h_(1) = q^1 h_(1) ,
    故
        Q = \oint q . nu ds = \int_0^1 q^1 h_(1) h_(2) dx^2 = \int_0^1 q^1 J dx^2 。
    床 p1 满足 z_2 ≡ 0，于是 t_2 = (0, 1, 0)、s^2 = x^2，积分变量即 x^2。
    """
    return _trapz(np.asarray(h_col) * np.asarray(u1_col) * np.asarray(J_col), ye)


def evaluate_field(net, ne_x=401, ne_y=41, t_list=None):
    r"""在细网格上评估若干时刻的场并计算诊断量。

    场量与诊断（全部按内蕴定义）
      h, u^1, u^2             网络输出（第 1 个输出即法向水深；本算例无 H_0 拟设）
      zeta = z + gamma h      自由水面高程
      J = sqrt(det G_sw)      面积元因子                                   [Eq.(4)]
      \bar h(x^1, t)          截面平均水深
      M(t)     = \int_U h J dx^1 dx^2                               总质量
      Q_out(t) = \int_{x^1 = X1} h u^1 J dx^2                       出口流量
      Q_s(t)   = 三截面 x^1 = 2.5 / 5.0 / 7.5 m 的流量
      x_shock  = argmax_{x^1 >= DAM_X} |d\bar h / d x^1|            坝后涌波锋位
      x_rare   = argmax_{x^1 <  DAM_X} |d\bar h / d x^1|            上游稀疏波位置
      d2max    = max |d_2 h|                                        应 ~ 0（解与 x^2 无关）

    无需参考解的守恒判据：连续方程 Eq.(14) 对域积分，用固壁（x^1 = 0, x^2 = 0,1 上
    q . nu = 0）与出流（x^1 = 10）边界，得**精确恒等式**
        dM/dt + Q_out = 0 .
    """
    if t_list is None:
        t_list = [0.0, 0.25 * T_END, 0.5 * T_END, 0.75 * T_END, T_END]
    t_list = [float(t) for t in t_list]
    xe = np.linspace(X0, X1, ne_x)
    ye = np.linspace(Y0, Y1, ne_y)
    Xg, Yg = np.meshgrid(xe, ye, indexing="ij")
    xy = np.stack([Xg.ravel(), Yg.ravel()], axis=1)
    dx = xe[1] - xe[0]
    dy = ye[1] - ye[0]
    # 梯形权重（与 q_section 的 _trapz 同口径）            # <<< FIX1 (P2-5) >>>
    _wx = np.full(xe.size, dx)
    _wx[0] *= 0.5
    _wx[-1] *= 0.5
    _wy = np.full(ye.size, dy)
    _wy[0] *= 0.5
    _wy[-1] *= 0.5
    W2 = np.outer(_wx, _wy)

    i_dam = int(np.argmin(np.abs(xe - DAM_X)))
    lo = slice(0, i_dam + 1)      # x^1 <= DAM_X（含坝线）
    hi = slice(i_dam, ne_x)       # x^1 >= DAM_X
    isect = [int(np.argmin(np.abs(xe - xs))) for xs in XSECT]

    out = {"x": xe, "y": ye, "t_list": np.array(t_list),
           "h": [], "u1": [], "u2": [], "zeta": [],
           "prof_h": [], "prof_zeta": [], "prof_u1": [],
           "mass": [], "q_out": [], "sect_Q": [],
           "x_shock": [], "x_rare": [], "d2max": []}

    for tv in t_list:
        X = torch.tensor(np.column_stack([xy, np.full(xy.shape[0], tv)]),
                         dtype=torch.float64)
        with torch.no_grad():
            h, u1, u2 = net(X)
            geo = geometry(X[:, 0], X[:, 1])
            sh = Xg.shape
            h_np = h.detach().cpu().numpy().reshape(sh)
            u1_np = u1.detach().cpu().numpy().reshape(sh)
            u2_np = u2.detach().cpu().numpy().reshape(sh)
            J_np = geo.J.detach().cpu().numpy().reshape(sh)
            zeta = (h * geo.N + geo.B).detach().cpu().numpy().reshape(sh)

        # 独立度量：本算例精确解与 x^2 无关 => d_2 h 应 ~ 0（需 autograd，单独算）
        Xg2 = X.clone().requires_grad_(True)
        hg, _, _ = net(Xg2)
        d2 = torch.autograd.grad(hg.sum(), Xg2, create_graph=False)[0][:, 1]
        d2max = float(d2.detach().abs().max())

        ph = profile_x(h_np, ye)
        dph = np.gradient(ph, xe)
        out["h"].append(h_np)
        out["u1"].append(u1_np)
        out["u2"].append(u2_np)
        out["zeta"].append(zeta)
        out["prof_h"].append(ph)
        out["prof_zeta"].append(profile_x(zeta, ye))
        out["prof_u1"].append(profile_x(u1_np, ye))
        out["mass"].append(float(np.sum(h_np * J_np * W2)))   # <<< FIX1 (P2-5b) >>>
        out["q_out"].append(q_section(h_np[-1, :], u1_np[-1, :], J_np[-1, :], ye))
        out["sect_Q"].append([q_section(h_np[i, :], u1_np[i, :], J_np[i, :], ye)
                              for i in isect])
        out["x_shock"].append(float(xe[hi.start + int(np.argmax(np.abs(dph[hi])))]))
        out["x_rare"].append(float(xe[int(np.argmax(np.abs(dph[lo])))]))
        out["d2max"].append(d2max)

    for k in ("h", "u1", "u2", "zeta", "prof_h", "prof_zeta", "prof_u1"):
        out[k] = np.array(out[k])
    for k in ("mass", "q_out", "x_shock", "x_rare", "d2max"):
        out[k] = np.array(out[k])
    out["sect_Q"] = np.array(out["sect_Q"])
    out["sect_x"] = np.array(XSECT)
    return out


def report_eval(ev, ev_ts=None):
    r"""打印评估报告（含无需参考解的质量-出流平衡判据）。"""
    print()
    print("=" * 78)
    print("  评估：抛物床平面溃坝（Bachini2020 §4.2 parabola case，无摩擦）")
    print("=" * 78)
    xe = ev["x"]
    i_d = int(np.argmin(np.abs(xe - DAM_X)))
    _, gam_d = bed_profile([DAM_X])
    gam_d = float(gam_d[0])
    h0, z0 = ev["h"][0], ev["zeta"][0]
    print("  初值自检（t = 0，取**训练后**的网络）：")
    print(f"    h     范围 = [{h0.min():.6f}, {h0.max():.6f}] m"
          f"     （目标 {H_DN:g} / {H_UP:g} m）")
    print(f"    zeta  范围 = [{z0.min():.6f}, {z0.max():.6f}] m"
          f"     （非常数 => 非静水态，符合 §4.2 的设计）")
    pz0 = ev["prof_zeta"][0]
    jump = float(pz0[i_d] - pz0[i_d + 1])
    print(f"    坝线处 zeta 跳跃 = {jump:.6f} m"
          f"   （闭式 gamma(DAM_X)*(H_UP-H_DN) = {gam_d * (H_UP - H_DN):.6f} m）")

    print()
    print("  总质量与出流（无需参考解的守恒判据  dM/dt + Q_out = 0）：")
    src = ev_ts if ev_ts is not None else ev
    tt = np.asarray(src["t_list"], dtype=float)
    M = np.asarray(src["mass"], dtype=float)
    Qo = np.asarray(src["q_out"], dtype=float)
    dMdt = np.gradient(M, tt) if len(tt) > 1 else np.zeros_like(M)
    print("      t [s]        M(t)            Q_out(t)        dM/dt + Q_out")
    for i in range(len(tt)):
        print(f"    {tt[i]:7.3f}   {M[i]:14.8f}   {Qo[i]:14.8f}   "
              f"{dMdt[i] + Qo[i]:+14.6e}")
    sc = max(np.max(np.abs(Qo)), 1e-12)
    print(f"    平衡残差 max|dM/dt + Q_out| = "
          f"{np.max(np.abs(dMdt + Qo)):.6e}   "
          f"（相对出口流量峰值 {np.max(np.abs(dMdt + Qo)) / sc:.3e}）")

    print()
    print(f"  截面流量 Q(t) = \\int_0^1 h u^1 J dx^2  [m^3/s]，截面 x^1 = "
          + ", ".join("%g" % s for s in XSECT) + " m：")
    print("      t [s]   " + "".join(f"   x^1={s:>5g}" for s in XSECT))
    for i, tv in enumerate(ev["t_list"]):
        print(f"    {tv:7.3f}   "
              + "".join(f"   {v:11.6f}" for v in ev["sect_Q"][i]))

    print()
    print("  锋位估计（截面平均剖面 |d\\bar h/d x^1| 的极大点）：")
    print("      t [s]     x_shock [m]    x_rare [m]")
    for i, tv in enumerate(ev["t_list"]):
        print(f"    {tv:7.3f}   {ev['x_shock'][i]:11.4f}   {ev['x_rare'][i]:11.4f}")
    hT = ev["h"][-1]
    print()
    print(f"  t = T 时 h 范围 = [{hT.min():.6f}, {hT.max():.6f}]")
    print(f"  y 无关性: max|d_2 h| = {np.max(ev['d2max']):.3e}（应 ~ 0，"
          f"本算例精确解与 x^2 无关）")
    print("  [参考解] 未提供：论文的 MAE / RMSE 须由内蕴有限体积解给出（§4.1）。")
    print("=" * 78)


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 11})
    return plt


def make_topography_plot(outdir):
    r"""床面 Eq.(22)、gamma(x^1)、初始法向水深台阶。"""
    plt = _mpl()
    xe = np.linspace(X0, X1, 601)
    z, gam = bed_profile(xe)
    h_ic = np.where(xe < DAM_X, H_UP, H_DN)
    zeta_ic = z + gam * h_ic

    fig, ax = plt.subplots(1, 3, figsize=(15.4, 4.2))
    ax[0].plot(xe, z, "k-", lw=1.8, label=r"bed $z$")
    ax[0].fill_between(xe, z, zeta_ic, color="tab:blue", alpha=0.25,
                       label=r"water column")
    ax[0].plot(xe, zeta_ic, color="tab:blue", lw=1.8,
               label=r"initial $\zeta=z+\gamma h$")
    ax[0].axvline(DAM_X, color="tab:red", ls="--", lw=1.3,
                  label=rf"dam $x^1={DAM_X:g}$ m")
    ax[0].set_xlabel(r"$x^1$ [m]")
    ax[0].set_ylabel(r"elevation [m]")
    ax[0].set_title(r"Bed Eq.(22) and initial free surface (normal depth)")
    ax[0].grid(True, ls=":", alpha=0.5)
    ax[0].legend(fontsize=8, loc="lower left")

    ax[1].plot(xe, gam, color="tab:green", lw=1.8)
    ax[1].axvline(DAM_X, color="tab:red", ls="--", lw=1.3)
    ax[1].set_xlabel(r"$x^1$ [m]")
    ax[1].set_ylabel(r"$\gamma$")
    ax[1].set_title(r"$\gamma = 1/\sqrt{1+|\nabla z|^2} \in [0.7809, 1]$")
    ax[1].grid(True, ls=":", alpha=0.5)

    ax[2].plot(xe, h_ic, color="tab:purple", lw=1.8)
    ax[2].axvline(DAM_X, color="tab:red", ls="--", lw=1.3)
    ax[2].set_xlabel(r"$x^1$ [m]")
    ax[2].set_ylabel(r"$h$  [m]")
    ax[2].set_title(rf"Initial normal depth: {H_UP:g} / {H_DN:g} m")
    ax[2].grid(True, ls=":", alpha=0.5)

    fig.tight_layout()
    fn = os.path.join(outdir, f"pinn_dam_parab_{BED}_topography.png")
    plt.savefig(fn, dpi=160, bbox_inches="tight")
    plt.close()
    return fn


def make_convergence_plot(hist, outdir):
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
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
    ax.set_title(rf"Convergence, parabola-case dam break, bed '{BED}'")
    ax.grid(True, which="both", ls=":", alpha=0.6)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fn = os.path.join(outdir, f"pinn_dam_parab_{BED}_convergence.png")
    plt.savefig(fn, dpi=160, bbox_inches="tight")
    plt.close()
    return fn


def make_snapshot_grid(ev, outdir):
    r"""上排平面视图 h(x^1,x^2,t)，下排纵向剖面（垂直放大 VE）。

    帧数多于 5 时自动分页（每页最多 5 帧）：第一页沿用原文件名
    ..._snapshots.png，其后为 ..._snapshots2.png / _snapshots3.png ...
    返回本函数写出的全部文件名（list）。

    <<< SHOCK_SNAP v2 >>>：默认快照表是 0.1 s 间隔的完整时刻（T_END = 1.5 s
    时为 16 帧），故分页是常态。
    """
    plt = _mpl()
    xe, ye, ts = ev["x"], ev["y"], ev["t_list"]
    nt = len(ts)
    z, _ = bed_profile(xe)
    hmin = min(a.min() for a in ev["h"])
    hmax = max(a.max() for a in ev["h"])
    per = 5
    pages = int(np.ceil(nt / per))
    made = []
    for pg in range(pages):
        idx = list(range(pg * per, min((pg + 1) * per, nt)))
        m = len(idx)
        fig, axes = plt.subplots(2, m, figsize=(3.6 * m, 7.6), squeeze=False)
        for j, k in enumerate(idx):
            tv = ts[k]
            ax = axes[0][j]
            im = ax.pcolormesh(xe, ye, ev["h"][k].T, cmap="viridis",
                               shading="auto", vmin=hmin, vmax=hmax)
            ax.axvline(DAM_X, color="w", ls="--", lw=1.3)
            ax.set_title(rf"$t={tv:g}$ s", fontsize=11)
            ax.set_xlabel(r"$x^1$ [m]")
            if j == 0:
                ax.set_ylabel(r"$x^2$ [m]")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03,
                         label=r"$h$ [m]" if j == m - 1 else None)

            ax = axes[1][j]
            ax.plot(xe, VE * z, color="0.35", lw=1.6,
                    label=r"bed $z$" if j == 0 else None)
            ax.plot(xe, VE * ev["prof_zeta"][k], color="tab:blue", lw=1.9,
                    label=r"$\zeta=z+\gamma h$" if j == 0 else None)
            ax.fill_between(xe, VE * z, VE * ev["prof_zeta"][k],
                            color="tab:blue", alpha=0.18)
            ax.axvline(DAM_X, color="tab:red", ls="--", lw=1.1)
            ax.set_xlabel(r"$x^1$ [m]")
            if j == 0:
                ax.set_ylabel(rf"elevation $\times${VE:g} [m]")
                ax.legend(fontsize=8, loc="upper right")
            ax.grid(True, ls=":", alpha=0.5)
        suffix = "" if pg == 0 else str(pg + 1)
        fig.suptitle(rf"Parabola-case dam break (bed Eq.(22), dam "
                     rf"$x^1={DAM_X:g}$ m, normal depth {H_UP:g}/{H_DN:g} m) — "
                     rf"vertical magnification $\times${VE:g}"
                     + ("" if pages == 1 else rf"  [page {pg + 1}/{pages}]"),
                     fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fn = os.path.join(outdir,
                          f"pinn_dam_parab_{BED}_snapshots{suffix}.png")
        plt.savefig(fn, dpi=160, bbox_inches="tight")
        plt.close()
        made.append(fn)
    return made


def make_single_time_plots(ev, outdir):
    r"""每个输出时刻单独两张图：平面视图 + 纵向剖面（垂直放大 2.0）。"""
    plt = _mpl()
    xe, ye, ts = ev["x"], ev["y"], ev["t_list"]
    z, _ = bed_profile(xe)
    hmin = min(a.min() for a in ev["h"])
    hmax = max(a.max() for a in ev["h"])
    made = []
    for k, tv in enumerate(ts):
        tag = _tslug(tv)
        fig, ax = plt.subplots(figsize=(7.6, 2.5))
        im = ax.pcolormesh(xe, ye, ev["h"][k].T, cmap="viridis", shading="auto",
                           vmin=hmin, vmax=hmax)
        ax.axvline(DAM_X, color="w", ls="--", lw=1.3)
        ax.set_xlabel(r"$x^1$ [m]")
        ax.set_ylabel(r"$x^2$ [m]")
        ax.set_title(rf"Parabola case: $h$ at $t={tv:g}$ s  "
                     rf"(dam $x^1={DAM_X:g}$ m)", fontsize=11)
        fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label=r"$h$ [m]")
        fig.tight_layout()
        f1 = os.path.join(outdir, f"pinn_dam_parab_{BED}_t{tag}_field.png")
        plt.savefig(f1, dpi=160, bbox_inches="tight")
        plt.close()

        fig, ax = plt.subplots(figsize=(7.6, 3.8))
        ax.plot(xe, VE * z, color="0.35", lw=1.7, label=r"bed $z$")
        ax.plot(xe, VE * ev["prof_zeta"][k], color="tab:blue", lw=2.0,
                label=r"free surface $\zeta=z+\gamma h$")
        ax.fill_between(xe, VE * z, VE * ev["prof_zeta"][k],
                        color="tab:blue", alpha=0.18)
        ax.axvline(DAM_X, color="tab:red", ls="--", lw=1.2,
                   label=rf"dam $x^1={DAM_X:g}$ m")
        ax.set_xlabel(r"$x^1$ [m]")
        ax.set_ylabel(rf"elevation $\times${VE:g} [m]")
        ax.set_title(rf"Parabola case: section at $t={tv:g}$ s  "
                     rf"(vertical magnification $\times${VE:g})", fontsize=11)
        ax.grid(True, ls=":", alpha=0.5)
        ax.legend(fontsize=9, loc="upper right")
        fig.tight_layout()
        f2 = os.path.join(outdir, f"pinn_dam_parab_{BED}_t{tag}_section.png")
        plt.savefig(f2, dpi=160, bbox_inches="tight")
        plt.close()
        made += [f1, f2]
    return made


def make_profile_plots(ev, outdir, ev_ts=None):
    r"""剖面演化 + 锋位随时间。"""
    plt = _mpl()
    xe, ts = ev["x"], ev["t_list"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.4, 4.6))
    for k, tv in enumerate(ts):
        a1.plot(xe, ev["prof_h"][k], lw=1.8, label=rf"$t={tv:g}$ s")
    a1.axvline(DAM_X, color="tab:red", ls="--", lw=1.2,
               label=rf"dam $x^1={DAM_X:g}$ m")
    a1.set_xlabel(r"$x^1$ [m]")
    a1.set_ylabel(r"section-averaged $\bar h$ [m]")
    a1.set_title(r"Depth profiles $\bar h(x^1,t)$")
    a1.grid(True, ls=":", alpha=0.6)
    a1.legend(fontsize=9)

    src = ev_ts if ev_ts is not None else ev
    a2.plot(src["t_list"], src["x_shock"], "s-", color="tab:red", lw=1.8, ms=5,
            label=r"$x_{\rm shock}$ (downstream, $x^1 \geq 2$)")
    a2.plot(src["t_list"], src["x_rare"], "o-", color="tab:blue", lw=1.8, ms=5,
            label=r"$x_{\rm rare}$ (upstream, $x^1 < 2$)")
    a2.axhline(DAM_X, color="k", ls="--", lw=1.0)
    a2.set_xlabel(r"$t$ [s]")
    a2.set_ylabel(r"position [m]")
    a2.set_title(r"Steepest-gradient positions vs time")
    a2.grid(True, ls=":", alpha=0.6)
    a2.legend(fontsize=9)
    fig.tight_layout()
    fn = os.path.join(outdir, f"pinn_dam_parab_{BED}_profiles.png")
    plt.savefig(fn, dpi=160, bbox_inches="tight")
    plt.close()
    return fn


def make_discharge_plots(ev_ts, outdir):
    r"""三个截面的流量 Q(t)。"""
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    for j, xs in enumerate(ev_ts["sect_x"]):
        ax.plot(ev_ts["t_list"], ev_ts["sect_Q"][:, j], lw=1.9,
                label=rf"$x^1={xs:g}$ m")
        ax.plot(ev_ts["t_list"], ev_ts["sect_Q"][:, j], "o", ms=3.5)
    ax.axhline(0.0, color="k", lw=0.8, ls="--")
    ax.set_xlabel(r"$t$ [s]")
    ax.set_ylabel(r"$Q$ [m$^3$/s]")
    ax.set_title(r"Sections discharge $Q=\int_0^1 h u^1 J\,dx^2$")
    ax.grid(True, ls=":", alpha=0.6)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fn = os.path.join(outdir, f"pinn_dam_parab_{BED}_discharge.png")
    plt.savefig(fn, dpi=160, bbox_inches="tight")
    plt.close()
    return fn


def make_mass_plots(ev_ts, outdir):
    r"""总质量、出口流量与质量-出流平衡残差。"""
    plt = _mpl()
    tt = np.asarray(ev_ts["t_list"], dtype=float)
    M = np.asarray(ev_ts["mass"], dtype=float)
    Qo = np.asarray(ev_ts["q_out"], dtype=float)
    dMdt = np.gradient(M, tt) if len(tt) > 1 else np.zeros_like(M)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.4, 4.6))
    a1.plot(tt, M, "-", color="tab:purple", lw=2.0, label=r"$M(t)$")
    a1.set_xlabel(r"$t$ [s]")
    a1.set_ylabel(r"$M=\int_U hJ\,dx^1dx^2$  [m$^3$]")
    a1.set_title(r"Total mass (decreases: outflow at $x^1=10$)")
    a1.grid(True, ls=":", alpha=0.6)
    a1.legend(fontsize=9)

    a2.plot(tt, dMdt, "-", color="tab:red", lw=1.9, label=r"$\mathrm{d}M/\mathrm{d}t$")
    a2.plot(tt, -Qo, "--", color="tab:blue", lw=1.9,
            label=r"$-Q_{\rm out}$ (continuum identity)")
    a2.set_xlabel(r"$t$ [s]")
    a2.set_ylabel(r"rate [m$^3$/s]")
    a2.set_title(r"Mass balance: $\mathrm{d}M/\mathrm{d}t + Q_{\rm out}=0$")
    a2.grid(True, ls=":", alpha=0.6)
    a2.legend(fontsize=9)
    fig.tight_layout()
    fn = os.path.join(outdir, f"pinn_dam_parab_{BED}_mass_balance.png")
    plt.savefig(fn, dpi=160, bbox_inches="tight")
    plt.close()
    return fn


# ============================================================================
# 11c. 时域混叠检测与质量-出流平衡核验              # <<< REFORM v1 (E1/E2) >>>
# ============================================================================
def _lpde_of(net, X, chunk=8192):
    r"""在给定点集上算 L_pde（分批，避免一次性二阶导爆显存）。

    chunk 与 N_INT 解耦：float64 下 6x128 网络的二阶导反向图每点约 75 kB，
    32768 点整批约 2.4 GB，而采样审计还要在 10 倍点集上再算一次。取 8192。
    分批只改变求和顺序，加权均值与原式数学等价。
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
    与训练配点上的 L_pde 比较：

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


def mass_balance_report(net, n_t=200, ne_x=201, ne_y=11, tag="", criterion=True):
    r"""在 t ~ U(0, T) 的 n_t 个**随机**时刻上核验质量-出流平衡。

    由 Eq.(14) 对域积分，用固壁（x^1 = X0、x^2 = Y0,Y1 上 q.n = 0）与出流
    （x^1 = X1）边界，得**精确恒等式**

        dM/dt + Q_out = 0 。

    只在少数固定时刻取样时，取样点可能恰好落在配点平面上；**随机时刻是非配点
    取样**，正是暴露时域混叠的量（E2P 上一轮 M(0) = 2.936 -> 1.5 s 时 0.2895，
    掉 90.1%，而各损失项全部"满足"）。判据：相对残差 < 5e-2。
    """
    xe = np.linspace(X0, X1, int(ne_x))
    ye = np.linspace(Y0, Y1, int(ne_y))
    Xg, Yg = np.meshgrid(xe, ye, indexing="ij")
    xy = np.stack([Xg.ravel(), Yg.ravel()], axis=1)
    area = (xe[1] - xe[0]) * (ye[1] - ye[0])
    # 梯形权重（与 evaluate_field 的 M(t)、q_section 同口径）
    #                                                     # <<< FIX1 (P2-5c) >>>
    _wx = np.full(xe.size, xe[1] - xe[0])
    _wx[0] *= 0.5
    _wx[-1] *= 0.5
    _wy = np.full(ye.size, ye[1] - ye[0])
    _wy[0] *= 0.5
    _wy[-1] *= 0.5
    Xout = np.column_stack([np.full(len(ye), X1), ye, np.zeros(len(ye))])

    _W2t = torch.tensor(np.outer(_wx, _wy).reshape(-1), dtype=torch.float64)
    #                                                     # <<< FIX1 (P2-5d) >>>

    def _M(t):
        X = torch.tensor(np.column_stack([xy, np.full(xy.shape[0], float(t))]),
                         dtype=torch.float64)
        with torch.no_grad():
            h, _, _ = net(X)
            geo = geometry(X[:, 0], X[:, 1])
            return float(torch.sum(h * geo.J * _W2t).detach())

    def _Qout(t):
        Xo = torch.tensor(Xout.copy(), dtype=torch.float64)
        Xo[:, 2] = float(t)
        with torch.no_grad():
            h, u1, _ = net(Xo)
            geo = geometry(Xo[:, 0], Xo[:, 1])
            return q_section(h.detach().cpu().numpy(), u1.detach().cpu().numpy(),
                             geo.J.detach().cpu().numpy(), ye)

    rs = np.random.RandomState(SEED + 4242)
    tv = np.sort(rs.uniform(0.05 * T_END, 0.95 * T_END, int(n_t)))
    hs = 1.0e-3
    Mp = np.array([_M(float(t) + hs) for t in tv])
    Mm = np.array([_M(float(t) - hs) for t in tv])
    Qs = np.array([_Qout(float(t)) for t in tv])
    dMdt = (Mp - Mm) / (2.0 * hs)
    res = dMdt + Qs
    scale = max(float(np.max(np.abs(Qs))), 1e-12)
    worst = float(np.max(np.abs(res)))
    rel = worst / scale
    print(f"  [{tag}] 随机时刻质量-出流平衡（{n_t} 个 t ~ U(0,{T_END:g})；"
          f"M 用 {ne_x}x{ne_y} 网格，Q_out 用 {ne_y} 点梯形）：")
    verdict = (("通过 ✅（< 5e-2）" if rel < 5e-2 else "失败 ❌（>= 5e-2）")
               if criterion else
               "—（训练前基线：初始网络无物理意义，不作判据）")
    print(f"  [{tag}]   M(0) = {_M(0.0):.8f}；"
          f"max|dM/dt + Q_out| = {worst:.6e}  （出口峰值 {scale:.6e}，"
          f"相对 {rel:.3e}）   " + verdict)
    return float(_M(0.0)), tv, res, rel


# ============================================================================
# 11d. 不同时刻的 3D 水面高度图                              # <<< SURF3D v1 >>>
# ============================================================================
def _surface_evo_panel(plt, ev, outdir, mode):
    r"""3D 演化多面板。mode = 'h'（法向水深）或 'zeta'（自由水面）。返回文件名。"""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    xe = np.asarray(ev["x"], dtype=float)
    ts = np.asarray(ev["t_list"], dtype=float)
    nt = len(ts)
    sx = max(1, len(xe) // 201)
    xd = xe[::sx]
    Xg, Yg = np.meshgrid(xd, np.array([Y0, Y1]), indexing="ij")
    z0 = VE * bed_profile(xe)[0][::sx]

    if mode == "h":
        F = np.asarray(ev["h"], dtype=float)[:, ::sx, 0]
        cmap = plt.get_cmap("Blues")
        vlo, vhi = float(F.min()), float(F.max())
        zlab = r"$h$ [m]"
        ttl = (r"Parabola-case dam break: normal depth $h$ at 0.1 s "
               r"intervals (bed Eq.(22), dam $x^1=%g$ m, normal depth %g/%g m)"
               % (DAM_X, H_UP, H_DN))
        fn_key = "surface3d_evolution_h"
    else:
        F = VE * np.asarray(ev["zeta"], dtype=float)[:, ::sx, 0]
        cmap = plt.get_cmap("viridis")
        vlo = float(min(F.min(), z0.min()))
        vhi = float(max(F.max(), z0.max()))
        zlab = rf"$\zeta\times{VE:g}$ [m]"
        ttl = (r"Parabola-case dam break: free surface $\zeta=z+\gamma h$ at "
               r"0.1 s intervals (vertical magnification $\times$%g; "
               r"cf. Bachini2020 Fig.4)" % VE)
        fn_key = "surface3d_evolution"
    if vhi - vlo < 1e-12:
        vhi = vlo + 1.0
    norm = plt.Normalize(vmin=vlo, vmax=vhi)

    ncol = min(4, nt)
    nrow = int(np.ceil(nt / ncol))
    fig = plt.figure(figsize=(3.6 * ncol, 3.3 * nrow))
    for k in range(nt):
        ax = fig.add_subplot(nrow, ncol, k + 1, projection="3d")
        Zg = np.repeat(F[k][:, None], 2, axis=1)
        ax.plot_surface(Xg, Yg, Zg, cmap=cmap, norm=norm,
                        rstride=1, cstride=1, linewidth=0.0,
                        antialiased=False, shade=False)
        if mode == "zeta":
            for yv_ in (Y0, Y1):
                quads = [[(xd[i], yv_, F[k][i]), (xd[i + 1], yv_, F[k][i + 1]),
                          (xd[i + 1], yv_, z0[i + 1]), (xd[i], yv_, z0[i])]
                         for i in range(len(xd) - 1)]
                ax.add_collection3d(Poly3DCollection(
                    quads, facecolor="#4c72b0", alpha=0.55, edgecolor="none"))
            ax.plot(xd, np.full_like(xd, Y0), z0, color="0.30", lw=1.2)
            ax.plot(xd, np.full_like(xd, Y1), z0, color="0.30", lw=1.2)
        ax.set_title(r"$t=%g$ s" % ts[k], fontsize=10, pad=-4)
        ax.set_xlabel(r"$x^1$ [m]", fontsize=8, labelpad=-6)
        ax.set_ylabel(r"$x^2$ [m]", fontsize=8, labelpad=-8)
        ax.set_zlabel("", fontsize=8)      # 量纲由色条承担，避免与刻度重叠
        ax.set_yticks([Y0, Y1])            # 解与 x^2 无关，只标端点避免拥挤
        ax.set_zlim(vlo, vhi)
        ax.tick_params(labelsize=7, pad=-2)
        ax.view_init(elev=24, azim=-62)
        try:
            ax.set_box_aspect((2.2, 1.0, 1.15))
        except Exception:
            pass
        ax.plot([DAM_X, DAM_X], [Y0, Y1], [vlo, vlo], color="tab:red",
                ls="--", lw=1.0)
    fig.subplots_adjust(right=0.92, wspace=0.02, hspace=0.04)
    cax = fig.add_axes([0.935, 0.15, 0.011, 0.70])
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                 label=zlab)
    fig.suptitle(ttl, fontsize=12)
    fn = os.path.join(outdir, f"pinn_dam_parab_{BED}_{fn_key}.png")
    plt.savefig(fn, dpi=150, bbox_inches="tight")
    plt.close()
    return fn


def make_surface_evolution_plot(ev, outdir):
    r"""不同时刻 3D 直观图（**两张**）。

      ..._surface3d_evolution_h.png : 法向水深 h（看清波的传播）
      ..._surface3d_evolution.png   : 自由水面 zeta（对齐 Bachini2020 Fig.4，
                                      垂直放大 VE 倍）

    h 与 zeta 分开画是必要的：本算例床面从 4.0 m 降到 0 m，而水深只有
    0.2~0.5 m，画在同一个 z 轴上时水深被床的落差淹没、看不见波；h 图把
    竖向尺度交给水深本身，波的传播就一目了然。
    返回 [h 图, zeta 图]。
    """
    plt = _mpl()
    return [_surface_evo_panel(plt, ev, outdir, "h"),
            _surface_evo_panel(plt, ev, outdir, "zeta")]


def make_surface_single_plots(ev, outdir):
    r"""每个时刻单独一张 3D 双栏图：左 = 法向水深 h（Tian 风格柱面），
    右 = 自由水面 zeta（垂直放大 VE 倍，Bachini 风格板块）。"""
    plt = _mpl()
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    xe = np.asarray(ev["x"], dtype=float)
    ts = np.asarray(ev["t_list"], dtype=float)
    hh = np.asarray(ev["h"], dtype=float)
    zeta = np.asarray(ev["zeta"], dtype=float)
    nt = len(ts)
    sx = max(1, len(xe) // 201)
    xd = xe[::sx]
    z0 = VE * bed_profile(xe)[0][::sx]
    Xg, Yg = np.meshgrid(xd, np.array([Y0, Y1]), indexing="ij")
    hlo = float(hh.min())
    hhi = float(hh.max())
    if hhi - hlo < 1e-12:
        hhi = hlo + 1.0

    written = []
    for k in range(nt):
        tstr = _tslug(ts[k])
        ZS = VE * zeta[k][::sx, 0]
        Hg = np.repeat(hh[k][::sx, 0][:, None], 2, axis=1)
        vlo = float(min(ZS.min(), z0.min()))
        vhi = float(max(ZS.max(), z0.max()))

        fig = plt.figure(figsize=(11.4, 4.6))
        ax1 = fig.add_subplot(121, projection="3d")
        ax1.plot_surface(Xg, Yg, Hg, cmap=plt.get_cmap("Blues"),
                         vmin=hlo, vmax=hhi, rstride=1, cstride=1,
                         linewidth=0.0, antialiased=False, shade=False)
        ax1.set_title(r"normal depth $h$ at $t=%g$ s" % ts[k], fontsize=10)
        ax1.set_xlabel(r"$x^1$ [m]", fontsize=8, labelpad=-6)
        ax1.set_ylabel(r"$x^2$ [m]", fontsize=8, labelpad=-6)
        ax1.set_zlabel(r"$h$ [m]", fontsize=8, labelpad=-6)
        ax1.set_zlim(hlo, hhi)
        ax1.tick_params(labelsize=7, pad=-2)
        ax1.view_init(elev=26, azim=-62)
        try:
            ax1.set_box_aspect((2.2, 1.0, 1.0))
        except Exception:
            pass
        # 每时刻的 3D 图补 colorbar（原先颜色仅为装饰、无量纲标注）
        #                                                 # <<< FIX1 (P2-4a) >>>
        fig.colorbar(plt.cm.ScalarMappable(
            norm=plt.Normalize(vmin=hlo, vmax=hhi), cmap=plt.get_cmap("Blues")),
            ax=ax1, shrink=0.62, pad=0.10, label=r"$h$ [m]")

        ax2 = fig.add_subplot(122, projection="3d")
        norm = plt.Normalize(vmin=vlo, vmax=vhi)
        Zg = np.repeat(ZS[:, None], 2, axis=1)
        ax2.plot_surface(Xg, Yg, Zg, cmap=plt.get_cmap("viridis"), norm=norm,
                         rstride=1, cstride=1, linewidth=0.0,
                         antialiased=False, shade=False)
        for yv_ in (Y0, Y1):
            quads = [[(xd[i], yv_, ZS[i]), (xd[i + 1], yv_, ZS[i + 1]),
                      (xd[i + 1], yv_, z0[i + 1]), (xd[i], yv_, z0[i])]
                     for i in range(len(xd) - 1)]
            ax2.add_collection3d(Poly3DCollection(
                quads, facecolor="#4c72b0", alpha=0.55, edgecolor="none"))
        ax2.plot(xd, np.full_like(xd, Y0), z0, color="0.30", lw=1.2)
        ax2.plot(xd, np.full_like(xd, Y1), z0, color="0.30", lw=1.2)
        ax2.set_title(r"free surface $\zeta$ at $t=%g$ s (mag. $\times$%g)"
                      % (ts[k], VE), fontsize=10)
        ax2.set_xlabel(r"$x^1$ [m]", fontsize=8, labelpad=-6)
        ax2.set_ylabel(r"$x^2$ [m]", fontsize=8, labelpad=-6)
        ax2.set_zlabel(rf"$\zeta\times{VE:g}$ [m]", fontsize=8, labelpad=-6)
        ax2.set_zlim(vlo, vhi)
        ax2.tick_params(labelsize=7, pad=-2)
        ax2.view_init(elev=24, azim=-62)
        try:
            ax2.set_box_aspect((2.2, 1.0, 1.0))
        except Exception:
            pass
        fig.colorbar(plt.cm.ScalarMappable(
            norm=norm, cmap=plt.get_cmap("viridis")),
            ax=ax2, shrink=0.62, pad=0.10,
            label=rf"$\zeta\times{VE:g}$ [m]")     # <<< FIX1 (P2-4b) >>>

        fig.suptitle(r"Parabola-case dam break: bed Eq.(22), dam $x^1=%g$ m, "
                     r"normal depth %g/%g m" % (DAM_X, H_UP, H_DN), fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fn = os.path.join(outdir,
                          f"pinn_dam_parab_{BED}_t{tstr}_surface3d.png")
        plt.savefig(fn, dpi=150, bbox_inches="tight")
        plt.close()
        written.append(fn)
    return written


# ============================================================================
# 12. 主程序
# ============================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="E2-P / Bachini2020 §4.2 parabola case: 抛物床平面溃坝 "
                    "ISWE-PINN（N-LAAF）")
    p.add_argument("--bed", choices=sorted(BED_JET), default="p1",
                   help="床：p1（Eq.(22)，本算例默认）/ flat / p2 / p3")
    p.add_argument("--domain", type=str, default=None,
                   help="域 x0,x1,y0,y1；缺省取该床在论文中的声明域")
    p.add_argument("--dam-x", type=float, default=None,
                   help="平面坝位置 x^1（默认 2.0 m）")
    p.add_argument("--tend", type=float, default=None, help="结束时刻 T")
    p.add_argument("--bc", choices=["mixed", "wall", "zgrad"], default="mixed",
                   help="边界条件：mixed（默认，固壁 + 自由出流）/ wall / zgrad")
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
                   help="输出时刻（逗号分隔）；缺省时自动取 0.1 s 间隔的完整时刻表")
    p.add_argument("--sampler", choices=["sobol", "grid"], default=None,
                   help="配点采样：sobol = 低差异（默认，消除时域混叠）/ "
                        "grid = 旧张量网格（A/B 对照）")            # <<< REFORM v1 >>>
    p.add_argument("--n-int", type=int, default=None,
                   help="内部配点数（默认 %d）" % N_INT)          # <<< REFORM v1 >>>
    p.add_argument("--n-bc-side", type=int, default=None,
                   help="每边界面配点数（默认 %d）" % N_BC_SIDE)   # <<< REFORM v1 >>>

    p.add_argument("--outdir", type=str, default=None, help="输出目录")
    p.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu",
                   help="计算设备：cpu(默认) / cuda / auto")
    return p.parse_args()


def apply_cli(args):
    """把命令行覆盖写回全局量（必须在建网络/配点之前调用）。"""
    global BED, X0, X1, Y0, Y1, XC, YC, DAM_X, T_END, BC_MODE, REBALANCE
    global N_SCALE, W_A, W_I, W_B, N_ITER, NX, NY, NT, M_BC
    global WIDTH, LAYERS, LBFGS_MAXITER, LBFGS_MAXFUN, SEED
    global T_SNAP
    global N_INT, N_IC, N_BC_SIDE, SAMPLER, W_PDE_M   # <<< REFORM v1 >>>
    global DEVICE


    BED = args.bed
    d = BED_DEFAULTS[BED]
    X0, X1, Y0, Y1 = d["domain"]
    DAM_X = d["dam_x"]
    if args.domain:
        X0, X1, Y0, Y1 = [float(v) for v in args.domain.split(",")]
    if args.dam_x is not None:
        DAM_X = float(args.dam_x)
    XC, YC = 0.5 * (X0 + X1), 0.5 * (Y0 + Y1)   # 仅用于网络输入归一化
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
    # ---- <<< REFORM v1 >>> 采样器 ----
    if getattr(args, "sampler", None) is not None:
        SAMPLER = str(args.sampler)
    if getattr(args, "n_int", None) is not None:
        N_INT = int(args.n_int)
    if getattr(args, "n_bc_side", None) is not None:
        N_BC_SIDE = int(args.n_bc_side)

    if args.seed is not None:
        SEED = int(args.seed)
    if args.snap is not None:
        # 仅在显式给出 --snap 时固定时刻表；否则 T_SNAP 保持 None，
        # 由 snap_times() 自动生成 0.1 s 间隔的完整时刻表  # <<< SHOCK_SNAP v2 >>>
        T_SNAP = [float(v) for v in args.snap.split(",") if v.strip()]
        T_SNAP = sorted(t for t in T_SNAP if -1e-12 <= t <= T_END + 1e-12)


    # ---- 设备选择（须在建网络/配点/审计之前生效）----
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
        NX, NY, NT, M_BC = 21, 16, 5, 16
        N_INT, N_IC, N_BC_SIDE = 4000, 512, 256            # <<< REFORM v1 >>>
        WIDTH, LAYERS = 48, 3
        LBFGS_MAXITER, LBFGS_MAXFUN = 50, 400



def header_report():
    print("=" * 78)
    print("  E2-P / Bachini2020 §4.2 parabola case：抛物床平面溃坝 ISWE-PINN（N-LAAF）")
    print("=" * 78)
    print(f"  床        : {BED}  ({BED_DEFAULTS[BED]['note']})")
    print(f"  域        : [{X0:g}, {X1:g}] x [{Y0:g}, {Y1:g}]")
    print(f"  初始条件  : h(法向) = {H_UP:g} (x^1 < {DAM_X:g}) / {H_DN:g} "
          f"(x^1 >= {DAM_X:g})；u = v = 0    [Bachini2020 §4.2]")
    print(f"  垂向厚度  : gamma h（本床上 gamma in [0.7809, 1]；h 为**法向**水深）")
    print(f"  时间区间  : [0, {T_END:g}] s；输出时刻 t = "
          + ", ".join("%g" % t for t in snap_times()) + " s")

    print(f"  边界      : {BC_MODE}（x^1={X0:g} 固壁 / x^1={X1:g} 自由出流 / "
          f"x^2={Y0:g},{Y1:g} 固壁）；重平衡: {'开' if REBALANCE else '关'}")
    print(f"  网络      : {LAYERS} 隐层 x {WIDTH}，N-LAAF (n = {N_SCALE:g})，"
          f"w_a = {W_A:g}       [Eq.(12) + Jagtap2020]")
    print(f"  拟设      : 网络**直接输出法向水深 h**（本算例无静水面，Eq.(13) 的"
          f"参考水位 H_0 无作用，已删去）")
    print(f"  损失      : {W_F:g} L_pde + {W_I:g} L_ic + {W_B:g} L_bc "
          f"+ {W_POS:g} L_pos + {W_A:g} S(a)      [Eq.(16) + Jagtap2020]")
    print(f"  配点      : sampler = {SAMPLER}"
          + (f"  (Nf = {N_INT}, NI = {N_IC}, Nb = {4 * N_BC_SIDE})"
             if SAMPLER != "grid" else
             f"  (张量网格 {NX}x{NY}x{NT} = {NX * NY * NT}；NX={NX} 使坝线 "
             f"x^1={DAM_X:g} 落在节点上)"))
    print(f"  损失权重  : L_pde 内 R1 : R2 : R3 = {W_PDE_M:g} : 1 : 1；"
          f"W_A(S(a)) = {W_A:g}" + ("（已移除 S(a)）" if W_A == 0.0 else ""))

    _dname = torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "CPU"
    print(f"  设备      : {DEVICE.type}  ({_dname})")
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

    with torch.device("cpu"):          # 权重初始化固定在 CPU 上：
        net = NLAFNet()                # cuda 张量与 cpu 张量用各自的 RNG，同一个
    # torch.manual_seed(SEED) 会给出**不同**的初始化序列。钉在 CPU 后，
    # --device cuda 与 --device cpu 从同一初始权重出发（见 DEVICE_EQUIV_probe.log）。
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
    mass_balance_report(net, tag="训练前", criterion=False)

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
        return torch.cat([q.reshape(-1) for q in ps]).detach().cpu().numpy().astype(
            np.float64)

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

    # ---- 采样充分性 / 混叠检测 + 质量-出流平衡（训练后）  # <<< REFORM v1 (E1/E2) >>>
    print()
    print("=" * 78)
    print("  采样充分性审计（时域混叠检测）与质量-出流平衡 —— 训练后")
    print("=" * 78)
    sampling_adequacy_report(net, Xp, tag="训练后")
    mass_balance_report(net, tag="训练后")

    # ---------------- 评估（输出时刻 + 时间序列）----------------
    t_snap = snap_times()
    ev_snap = evaluate_field(net, t_list=t_snap)

    ev_ts = evaluate_field(net, t_list=list(np.linspace(0.0, T_END, 31)))
    report_eval(ev_snap, ev_ts)

    tag = f"pinn_dam_parab_{BED}"
    np.savez(os.path.join(outdir, f"{tag}_snap.npz"),
             x=ev_snap["x"], y=ev_snap["y"], t_snap=ev_snap["t_list"],
             h_snap=ev_snap["h"], zeta_snap=ev_snap["zeta"],
             u1_snap=ev_snap["u1"], u2_snap=ev_snap["u2"],
             prof_x=ev_snap["x"], prof_h_snap=ev_snap["prof_h"],
             prof_zeta_snap=ev_snap["prof_zeta"],
             sect_x=ev_snap["sect_x"], sect_Q_snap=ev_snap["sect_Q"],
             mass_snap=ev_snap["mass"], q_out_snap=ev_snap["q_out"],
             x_shock_snap=ev_snap["x_shock"], x_rare_snap=ev_snap["x_rare"],
             cfg=np.array([X0, X1, Y0, Y1, DAM_X, T_END, H_UP, H_DN,
                           N_SCALE, W_A, W_I, W_B, W_F, W_POS]))
    print(f"  数据: {tag}_snap.npz  (输出时刻 t = "
          + ", ".join("%g" % t for t in t_snap) + " s)")

    np.savez(os.path.join(outdir, f"{tag}.npz"),
             x=ev_ts["x"], y=ev_ts["y"], t_list=ev_ts["t_list"],
             h=ev_ts["h"], u1=ev_ts["u1"], u2=ev_ts["u2"],
             zeta=ev_ts["zeta"], prof_h=ev_ts["prof_h"],
             prof_zeta=ev_ts["prof_zeta"], sect_x=ev_ts["sect_x"],
             sect_Q=ev_ts["sect_Q"], mass=ev_ts["mass"], q_out=ev_ts["q_out"],
             x_shock=ev_ts["x_shock"], x_rare=ev_ts["x_rare"],
             d2max=ev_ts["d2max"],
             hist_it=np.array(hist["it"]), hist_total=np.array(hist["total"]),
             hist_pde=np.array(hist["pde"]), hist_ic=np.array(hist["ic"]),
             hist_bc=np.array(hist["bc"]),
             hist_pos=np.array(hist["pos"]),      # <<< FIX1 (P2-1) >>>
             hist_sa=np.array(hist["sa"]),
             cfg=np.array([X0, X1, Y0, Y1, DAM_X, T_END, H_UP, H_DN,
                           N_SCALE, W_A, W_I, W_B, W_F, W_POS]))
    print(f"  数据: {tag}.npz")

    if not args.no_plot:
        files = [make_topography_plot(outdir),
                 make_convergence_plot(hist, outdir)]
        files += make_snapshot_grid(ev_snap, outdir)
        files += [make_profile_plots(ev_snap, outdir, ev_ts),
                  make_discharge_plots(ev_ts, outdir),
                  make_mass_plots(ev_ts, outdir)]
        files += make_single_time_plots(ev_snap, outdir)
        files += make_surface_evolution_plot(ev_snap, outdir)   # <<< SURF3D v1 >>>
        files += make_surface_single_plots(ev_snap, outdir)     # <<< SURF3D v1 >>>

        print(f"  图件（{len(files)} 张）:")
        for f in files:
            print("    " + os.path.basename(f))
    print("=" * 78)
    print("  全部完成。")
    print("=" * 78)


if __name__ == "__main__":
    main()
