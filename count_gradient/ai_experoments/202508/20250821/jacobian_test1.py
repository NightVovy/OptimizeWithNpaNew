import numpy as np
import math
from itertools import product

# ------------------------------
# 用户给定：算符与参数->期望值
# ------------------------------
X = np.array([[0.0, 1.0], [1.0, 0.0]])
Z = np.array([[1.0, 0.0], [0.0, -1.0]])
I = np.eye(2)


def calculate_parameters(theta, a0, a1, b0, b1):
    """Compute A0,A1,B0,B1,E00,E01,E10,E11 from (theta,a0,a1,b0,b1) using user's construction.
    Returns dict with keys ['A0','A1','B0','B1','E00','E01','E10','E11'] and internals.
    """
    cos2theta = np.cos(2 * theta)
    sin2theta = np.sin(2 * theta)

    # 旋转角 mu = arctan(sin 2theta)
    mu = np.arctan(sin2theta)

    # 旋转后的态 |psi_r>
    psi_r = np.array([
        0.5 * (np.cos(theta) * (1 + np.cos(mu)) + np.sin(theta) * (1 - np.cos(mu))),
        0.5 * (np.cos(theta) - np.sin(theta)) * np.sin(mu),
        0.5 * (np.cos(theta) - np.sin(theta)) * np.sin(mu),
        0.5 * (np.cos(theta) * (1 - np.cos(mu)) + np.sin(theta) * (1 + np.cos(mu)))
    ])

    mea_A0 = np.cos(a0) * Z + np.sin(a0) * X
    mea_A1 = np.cos(a1) * Z + np.sin(a1) * X
    mea_B0 = np.cos(b0) * Z + np.sin(b0) * X
    mea_B1 = np.cos(b1) * Z + np.sin(b1) * X

    def tensor_op(op1, op2):
        return np.kron(op1, op2)

    def expectation(op):
        return float(np.dot(psi_r.conj(), np.dot(op, psi_r)).real)

    E00 = expectation(tensor_op(mea_A0, mea_B0))
    E01 = expectation(tensor_op(mea_A0, mea_B1))
    E10 = expectation(tensor_op(mea_A1, mea_B0))
    E11 = expectation(tensor_op(mea_A1, mea_B1))

    A0 = expectation(tensor_op(mea_A0, I))
    A1 = expectation(tensor_op(mea_A1, I))
    B0 = expectation(tensor_op(I, mea_B0))
    B1 = expectation(tensor_op(I, mea_B1))

    return {
        'A0': A0, 'A1': A1, 'B0': B0, 'B1': B1,
        'E00': E00, 'E01': E01, 'E10': E10, 'E11': E11,
        'mea_A0': mea_A0, 'mea_A1': mea_A1,
        'mea_B0': mea_B0, 'mea_B1': mea_B1,
        'b0': b0, 'b1': b1
    }


def params_to_x(theta, a0, a1, b0, b1):
    """Map (theta,a0,a1,b0,b1) -> 8D point x=[A0,A1,B0,B1,E00,E01,E10,E11]."""
    P = calculate_parameters(theta, a0, a1, b0, b1)
    return np.array([P['A0'], P['A1'], P['B0'], P['B1'], P['E00'], P['E01'], P['E10'], P['E11']], dtype=float)


# ------------------------------
# 4 条等式：对 (s,t) ∈ {±1}×{±1}
# f_{s,t}(p) := asin((E00+sB0)/(1+sA0)) + asin((E10+tB0)/(1+tA1)) - asin((E01+sB1)/(1+sA0)) + asin((E11+tB1)/(1+tA1)) - pi = 0
# ------------------------------

ST_ORDER = [(1, 1), (1, -1), (-1, 1), (-1, -1)]


def verify_formula_from_params(theta, a0, a1, b0, b1):
    P = calculate_parameters(theta, a0, a1, b0, b1)
    A0, A1 = P['A0'], P['A1']
    B0, B1 = P['B0'], P['B1']
    E00, E01 = P['E00'], P['E01']
    E10, E11 = P['E10'], P['E11']

    vals = []
    for s, t in ST_ORDER:
        denom0 = 1 + s * A0
        denoma1 = 1 + t * A1
        if abs(denom0) < 1e-12 or abs(denoma1) < 1e-12:
            vals.append(np.nan)
            continue
        sA0B0 = (E00 + s * B0) / denom0
        tA1B0 = (E10 + t * B0) / denoma1
        sA0B1 = (E01 + s * B1) / denom0
        tA1B1 = (E11 + t * B1) / denoma1
        # 域检查
        if max(abs(sA0B0), abs(tA1B0), abs(sA0B1), abs(tA1B1)) > 1 + 1e-10:
            vals.append(np.nan)
            continue
        value = (math.asin(sA0B0) + math.asin(tA1B0) - math.asin(sA0B1) + math.asin(tA1B1) - math.pi)
        vals.append(value)
    return np.array(vals, dtype=float)


def F_vec(theta, a0, a1, b0, b1):
    """Return 4-vector [f_{1,1}, f_{1,-1}, f_{-1,1}, f_{-1,-1}] evaluated at p."""
    return verify_formula_from_params(theta, a0, a1, b0, b1)


# ------------------------------
# 数值雅可比（中央差分）
# ------------------------------

def numerical_jacobian(func, p, h_abs=1e-7, h_rel=1e-7):
    """Compute Jacobian J_{ij} = d func_i / d p_j via central differences.
    func: R^5 -> R^m returns 1D array of length m.
    p: length-5 array (theta,a0,a1,b0,b1).
    """
    p = np.array(p, dtype=float)
    f0 = func(*p)
    m = f0.size
    n = p.size
    J = np.zeros((m, n), dtype=float)
    for j in range(n):
        step = h_abs + h_rel * max(1.0, abs(p[j]))
        pj_plus = p.copy(); pj_plus[j] += step
        pj_minus = p.copy(); pj_minus[j] -= step
        f_plus = func(*pj_plus)
        f_minus = func(*pj_minus)
        # 处理 NaN（可能来自 arcsin 域或分母），退化为前向差分
        mask = np.isfinite(f_plus) & np.isfinite(f_minus)
        if not np.all(mask):
            pj_plus2 = p.copy(); pj_plus2[j] += step
            f_plus2 = func(*pj_plus2)
            grad = (f_plus2 - f0) / step
        else:
            grad = (f_plus - f_minus) / (2.0 * step)
        J[:, j] = grad
    return J


def numerical_jacobian_x(p, h_abs=1e-7, h_rel=1e-7):
    """Compute X_p = d x / d p \in R^{8x5} with central differences on params_to_x."""
    p = np.array(p, dtype=float)
    x0 = params_to_x(*p)
    m = x0.size  # 8
    n = p.size   # 5
    J = np.zeros((m, n), dtype=float)
    for j in range(n):
        step = h_abs + h_rel * max(1.0, abs(p[j]))
        pj_plus = p.copy(); pj_plus[j] += step
        pj_minus = p.copy(); pj_minus[j] -= step
        x_plus = params_to_x(*pj_plus)
        x_minus = params_to_x(*pj_minus)
        grad = (x_plus - x_minus) / (2.0 * step)
        J[:, j] = grad
    return J


def stable_rank(A, eps=None):
    """Numerical rank via SVD with tolerance akin to MATLAB: tol = max(m,n)*eps*sigma_max."""
    if eps is None:
        eps = np.finfo(float).eps
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    tol = max(A.shape) * eps * (s[0] if s.size else 0.0)
    r = int(np.sum(s > tol))
    return r, s, tol


# ------------------------------
# 主流程：在给定参数点判定“光滑/非光滑”
# ------------------------------

def check_smoothness(theta, a0, a1, b0, b1, verbose=True):
    p = np.array([theta, a0, a1, b0, b1], dtype=float)

    # 1) 验证 4 条等式是否满足（应接近 0）
    residuals = F_vec(*p)

    # 2) 计算 J_p = dF/dp (4x5)
    Jp = numerical_jacobian(F_vec, p)
    r_Jp, s_Jp, tol_Jp = stable_rank(Jp)

    # 3) 计算 X_p = dx/dp (8x5)
    Xp = numerical_jacobian_x(p)
    r_Xp, s_Xp, tol_Xp = stable_rank(Xp)

    # 4) 由 J = F_x * X_p 推回 F_x ≈ J * X_p^+
    #    若 X_p 满列秩（秩=5），可用伪逆求 F_x；否则给出警告并仍计算最小二乘意义下的近似。
    Xp_pinv = np.linalg.pinv(Xp, rcond=1e-12)
    Fx = Jp @ Xp_pinv   # 形状 4x8
    r_Fx, s_Fx, tol_Fx = stable_rank(Fx)

    # 5) 判定逻辑（凸边界的法向维数=span{∇_x f_i} 的维数=r_Fx）
    #    r_Fx = 1 -> 单一法向（光滑）； r_Fx > 1 -> 多法向（非光滑/折角/尖点）。
    verdict = "光滑 (唯一法向)" if r_Fx == 1 else (
        "非光滑 (多法向)" if r_Fx > 1 else "无法判定")

    if verbose:
        print("=== 量子集合边界光滑性 Jacobian 判定 ===")
        print(f"参数 p = [theta, a0, a1, b0, b1] = {p}")
        print("-- 等式残差 F(p) (应接近 0)：\n", residuals)
        print("-- J_p = dF/dp 形状:", Jp.shape)
        print("   rank(J_p)=", r_Jp, " singular values:", s_Jp, " tol=", tol_Jp)
        print("-- X_p = dx/dp 形状:", Xp.shape)
        print("   rank(X_p)=", r_Xp, " singular values:", s_Xp, " tol=", tol_Xp)
        print("-- F_x ≈ J_p * pinv(X_p) 形状:", Fx.shape)
        print("   rank(F_x)=", r_Fx, " singular values:", s_Fx, " tol=", tol_Fx)
        if r_Xp < 5:
            print("[警告] 参数化在该点秩亏 (rank(X_p) < 5)。法向判定可能不稳定；建议更小步长或解析梯度。")
        print("== 结论 ==>", verdict)

    return {
        'residuals': residuals,
        'Jp': Jp, 'rank_Jp': r_Jp, 'sv_Jp': s_Jp, 'tol_Jp': tol_Jp,
        'Xp': Xp, 'rank_Xp': r_Xp, 'sv_Xp': s_Xp, 'tol_Xp': tol_Xp,
        'Fx': Fx, 'rank_Fx': r_Fx, 'sv_Fx': s_Fx, 'tol_Fx': tol_Fx,
        'verdict': verdict
    }


# ------------------------------
# 使用示例：请把你的 (theta,a0,a1,b0,b1) 填到下面，然后运行。
# ------------------------------
if __name__ == "__main__":
    # TODO: 替换为你的真实数值 (弧度)
    theta = np.pi / 6

    sin2theta = np.sin(2 * theta)
    # Calculate mu
    mu = np.arctan(sin2theta)

    a0 = 0
    a1 = 2 * mu
    b0 = mu
    b1 = np.pi / 2 + mu

    out = check_smoothness(theta, a0, a1, b0, b1, verbose=True)
