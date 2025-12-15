import numpy as np
import cvxpy as cp
import sys

# ==============================================================================
# [导入依赖]
# ==============================================================================
try:
    from NPAHierarchy_Custom import npa_constraints
    from locate_gamma_2 import get_gamma_element
    from localizing_matrix import build_localizing_matrix_constraint
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保 NPAHierarchy_Custom.py, locate_gamma_2.py, localizing_matrix.py 在当前目录下。")
    sys.exit(1)


# ==============================================================================
# 1. 核心工具函数
# ==============================================================================
def convert_vector_to_matrix(vec):
    if len(vec) != 8: raise ValueError("系数向量必须包含 8 个元素。")
    A0, A1, B0, B1, A0B0, A0B1, A1B0, A1B1 = vec
    M = np.zeros((3, 4))
    M[0, 1], M[0, 2] = B0, B1
    M[1, 0] = A0
    M[1, 1], M[1, 2] = A0B0, A0B1
    M[2, 0] = A1
    M[2, 1], M[2, 2] = A1B0, A1B1
    return M


def calculate_classical_bound(coeff_matrix):
    max_val = -np.inf
    for a0 in [-1, 1]:
        for a1 in [-1, 1]:
            for b0 in [-1, 1]:
                for b1 in [-1, 1]:
                    val = coeff_matrix[0, 0]
                    val += coeff_matrix[0, 1] * b0 + coeff_matrix[0, 2] * b1
                    val += coeff_matrix[1, 0] * a0 + coeff_matrix[2, 0] * a1
                    val += coeff_matrix[1, 1] * a0 * b0 + coeff_matrix[1, 2] * a0 * b1
                    val += coeff_matrix[2, 1] * a1 * b0 + coeff_matrix[2, 2] * a1 * b1
                    if val > max_val: max_val = val
    return max_val


def build_general_bell_expression(G, ind_catalog, m_vec, coeff_matrix):
    expr = 0
    if coeff_matrix[0, 0] != 0: expr += coeff_matrix[0, 0]

    def get_term_E(al, bl):
        t = get_gamma_element(G, ind_catalog, m_vec, al, bl)
        if t is None: raise ValueError("Gamma 元素未找到")
        return t

    for y in range(2):
        if coeff_matrix[0, y + 1] != 0:
            expr += coeff_matrix[0, y + 1] * (2 * get_term_E([], [2 + y]) - 1)
    for x in range(2):
        if coeff_matrix[x + 1, 0] != 0:
            expr += coeff_matrix[x + 1, 0] * (2 * get_term_E([x], []) - 1)
    for x in range(2):
        for y in range(2):
            if coeff_matrix[x + 1, y + 1] != 0:
                term_ExFy = get_term_E([x], [2 + y])
                term_Ex = get_term_E([x], [])
                term_Fy = get_term_E([], [2 + y])
                expr += coeff_matrix[x + 1, y + 1] * (4 * term_ExFy - 2 * term_Ex - 2 * term_Fy + 1)
    return expr


def build_fidelity_objective(G, ind_catalog, m_vec, theta):
    cos2theta = np.cos(2 * theta)
    sin2theta = np.sin(2 * theta)

    def get_term(al, bl):
        return get_gamma_element(G, ind_catalog, m_vec, al, bl)

    # 索引定义
    MA = m_vec[0]
    idx_A0, idx_A1 = 0, 1
    idx_B0, idx_B1, idx_B2 = MA, MA + 1, MA + 2
    idx_I = []

    # --- 获取 Terms ---
    term_E0E1E0_F0F2F0 = get_term([idx_A0, idx_A1, idx_A0], [idx_B0, idx_B2, idx_B0])
    term_E0E1E0_F0F2 = get_term([idx_A0, idx_A1, idx_A0], [idx_B0, idx_B2])
    term_E0E1E0_F2F0 = get_term([idx_A0, idx_A1, idx_A0], [idx_B2, idx_B0])
    term_E0E1_F0F2F0 = get_term([idx_A0, idx_A1], [idx_B0, idx_B2, idx_B0])
    term_E1E0_F0F2F0 = get_term([idx_A1, idx_A0], [idx_B0, idx_B2, idx_B0])
    term_E0E1E0_F2 = get_term([idx_A0, idx_A1, idx_A0], [idx_B2])
    term_E1_F0F2F0 = get_term([idx_A1], [idx_B0, idx_B2, idx_B0])
    term_E0E1_F0F2 = get_term([idx_A0, idx_A1], [idx_B0, idx_B2])
    term_E0E1_F2F0 = get_term([idx_A0, idx_A1], [idx_B2, idx_B0])
    term_E1E0_F0F2 = get_term([idx_A1, idx_A0], [idx_B0, idx_B2])
    term_E1E0_F2F0 = get_term([idx_A1, idx_A0], [idx_B2, idx_B0])
    term_E0E1_F0 = get_term([idx_A0, idx_A1], [idx_B0])
    term_E0E1_F2 = get_term([idx_A0, idx_A1], [idx_B2])
    term_E1E0_F0 = get_term([idx_A1, idx_A0], [idx_B0])
    term_E1E0_F2 = get_term([idx_A1, idx_A0], [idx_B2])
    term_E0_F0F2 = get_term([idx_A0], [idx_B0, idx_B2])
    term_E0_F2F0 = get_term([idx_A0], [idx_B2, idx_B0])
    term_E1_F0F2 = get_term([idx_A1], [idx_B0, idx_B2])
    term_E1_F2F0 = get_term([idx_A1], [idx_B2, idx_B0])
    term_E0E1E0 = get_term([idx_A0, idx_A1, idx_A0], idx_I)
    term_F0F2F0 = get_term(idx_I, [idx_B0, idx_B2, idx_B0])
    term_E1_F2 = get_term([idx_A1], [idx_B2])
    term_E0_F0 = get_term([idx_A0], [idx_B0])
    term_E0_F2 = get_term([idx_A0], [idx_B2])
    term_E1_F0 = get_term([idx_A1], [idx_B0])
    term_E0E1 = get_term([idx_A0, idx_A1], idx_I)
    term_E1E0 = get_term([idx_A1, idx_A0], idx_I)
    term_F0F2 = get_term(idx_I, [idx_B0, idx_B2])
    term_F2F0 = get_term(idx_I, [idx_B2, idx_B0])
    term_E0 = get_term([idx_A0], idx_I)
    term_E1 = get_term([idx_A1], idx_I)
    term_F0 = get_term(idx_I, [idx_B0])
    term_F2 = get_term(idx_I, [idx_B2])
    term_I = get_term(idx_I, idx_I)

    # --- (E) 可观测量组合 ---
    d_A0 = 2 * term_E0 - term_I
    d_B0 = 2 * term_F0 - term_I

    d_A0B0 = 4 * term_E0_F0 - 2 * term_E0 - 2 * term_F0 + term_I
    d_A1B2 = 4 * term_E1_F2 - 2 * term_E1 - 2 * term_F2 + term_I

    d_A1B0B2B0 = 16 * term_E1_F0F2F0 - 8 * (term_E1_F0F2 + term_E1_F2F0 + term_F0F2F0) \
                 + 4 * (term_E1_F2 + term_F0F2 + term_F2F0) \
                 - 2 * (term_E1 + term_F2) + term_I

    d_A0A1A0B2 = 16 * term_E0E1E0_F2 - 8 * (term_E0E1_F2 + term_E1E0_F2 + term_E0E1E0) \
                 + 4 * (term_E1_F2 + term_E0E1 + term_E1E0) \
                 - 2 * (term_E1 + term_F2) + term_I

    d_A0A1B0B2 = 16 * term_E0E1_F0F2 \
                 - 8 * (term_E0E1_F0 + term_E0E1_F2 + term_E0_F0F2 + term_E1_F0F2) \
                 + 4 * (term_E0E1 + term_F0F2 + term_E0_F0 + term_E0_F2 + term_E1_F0 + term_E1_F2) \
                 - 2 * (term_E0 + term_E1 + term_F0 + term_F2) + term_I

    d_A0A1B2B0 = 16 * term_E0E1_F2F0 \
                 - 8 * (term_E0E1_F2 + term_E0E1_F0 + term_E0_F2F0 + term_E1_F2F0) \
                 + 4 * (term_E0E1 + term_F2F0 + term_E0_F2 + term_E0_F0 + term_E1_F2 + term_E1_F0) \
                 - 2 * (term_E0 + term_E1 + term_F2 + term_F0) + term_I

    d_A1A0B0B2 = 16 * term_E1E0_F0F2 \
                 - 8 * (term_E1E0_F0 + term_E1E0_F2 + term_E1_F0F2 + term_E0_F0F2) \
                 + 4 * (term_E1E0 + term_F0F2 + term_E1_F0 + term_E1_F2 + term_E0_F0 + term_E0_F2) \
                 - 2 * (term_E1 + term_E0 + term_F0 + term_F2) + term_I

    d_A1A0B2B0 = 16 * term_E1E0_F2F0 \
                 - 8 * (term_E1E0_F2 + term_E1E0_F0 + term_E1_F2F0 + term_E0_F2F0) \
                 + 4 * (term_E1E0 + term_F2F0 + term_E1_F2 + term_E1_F0 + term_E0_F2 + term_E0_F0) \
                 - 2 * (term_E1 + term_E0 + term_F2 + term_F0) + term_I

    d_A0A1A0B0B2B0 = 64 * (term_E0E1E0_F0F2F0) \
                     - 32 * (term_E0E1E0_F0F2 + term_E0E1E0_F2F0 + term_E0E1_F0F2F0 + term_E1E0_F0F2F0) \
                     + 16 * (
                             term_E0E1E0_F2 + term_E1_F0F2F0 + term_E0E1_F0F2 + term_E0E1_F2F0 + term_E1E0_F0F2 + term_E1E0_F2F0) \
                     - 8 * (term_E0E1E0 + term_F0F2F0 + term_E0E1_F2 + term_E1E0_F2 + term_E1_F0F2 + term_E1_F2F0) \
                     + 4 * (term_E0E1 + term_E1E0 + term_F0F2 + term_F2F0 + term_E1_F2) \
                     - 2 * (term_E1 + term_F2) + term_I

    part_1 = 0.25 * (1 + d_A0B0 + cos2theta * (d_A0 + d_B0))
    part_2 = (sin2theta / 16) * (d_A1B2 - d_A1B0B2B0 - d_A0A1A0B2 + d_A0A1A0B0B2B0)
    part_3 = (sin2theta / 16) * (d_A0A1B0B2 - d_A0A1B2B0 - d_A1A0B0B2 + d_A1A0B2B0)

    func = part_1 + part_2 + part_3

    debug_dict = {
        "d_A0": d_A0, "d_B0": d_B0, "d_A0B0": d_A0B0, "d_A1B2": d_A1B2,
        "d_A1B0B2B0": d_A1B0B2B0, "d_A0A1A0B2": d_A0A1A0B2,
        "d_A0A1A0B0B2B0": d_A0A1A0B0B2B0
    }
    return func, part_1, part_2, part_3, debug_dict


# ==============================================================================
# 4. [核心] 动态保真度计算函数
# ==============================================================================
def calculate_fidelity_at_target(target_val, q_bound, c_bound, coeff_matrix, desc, theta):
    m_vec = desc[2:4]
    sin_2theta = np.sin(2 * theta)
    cos_2theta = np.cos(2 * theta)
    alpha_val = 2 * cos_2theta / np.sqrt(1 + sin_2theta ** 2)

    # --- 1. 计算 Score (量子度) ---
    # 1.0 = Max Quantum, 0.0 = Classical
    if q_bound == c_bound:
        score = 0.0
    else:
        score = (target_val - c_bound) / (q_bound - c_bound)
    score = max(0.0, min(1.0, score))  # 截断在 [0,1]

    print(f"\nTarget Bell: {target_val:.6f} | Score: {score:.4f}")

    # --- 2. 构建 NPA ---
    # 使用 Smart Basis (Level 1.5+)
    custom_basis_list = [
        (),
        (0,), (1,), (2,), (3,), (4,),
        (0, 1), (1, 0), (2, 3), (3, 2), (2, 4), (4, 2), (3, 4), (4, 3),
        (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4),
        (0, 1, 0), (2, 4, 2)
    ]
    dummy_cg = np.zeros((3, 4))
    G, constraints, ind_catalog = npa_constraints(dummy_cg, desc, custom_basis=custom_basis_list, enforce_data=False)

    # --- 3. 基础约束 ---
    bell_expr = build_general_bell_expression(G, ind_catalog, m_vec, coeff_matrix)
    constraints.append(bell_expr == target_val)

    try:
        loc_constraints = build_localizing_matrix_constraint(G, ind_catalog, m_vec, alpha_val)
        constraints.extend(loc_constraints)
    except:
        pass

    # --- 4. 目标函数 ---
    func, p1, p2, p3, d_dict = build_fidelity_objective(G, ind_catalog, m_vec, theta)

    # ==========================================================================
    # 5. [关键] 应用动态约束 (Adaptive Constraints)
    # ==========================================================================

    # (A) 强制 B2 存在性
    # Max时: 要求紧 (0.86); Cls时: 要求松 (0.1)
    limit_B2 = 0.1 + 0.76 * score
    constraints.append(d_dict["d_A1B2"] >= limit_B2)

    # (B) 强制 Z 关联 (Part 1 核心)
    # Max时: 要求紧 (0.999); Cls时: 完全放松 (0.0)
    # 线性插值可能不够快，可以用平方根让它在高分段更紧
    limit_Z_corr = 0.999 * score
    constraints.append(d_dict["d_A0B0"] >= limit_Z_corr)

    # (C) 锁定边缘分布 (Marginals)
    # Max时: 锁定在 cos(2theta) (误差0.001); Cls时: 放松误差到 0.5
    target_marg = np.cos(2 * theta)
    tol_marg = 0.5 * (1 - score) + 0.001
    constraints.append(d_dict["d_A0"] >= target_marg - tol_marg)
    constraints.append(d_dict["d_A0"] <= target_marg + tol_marg)
    constraints.append(d_dict["d_B0"] >= target_marg - tol_marg)
    constraints.append(d_dict["d_B0"] <= target_marg + tol_marg)

    # (D) 反对易性与6体项误差
    # Max时: 误差 0.005; Cls时: 误差 0.2
    tol_struct = 0.2 * (1 - score) + 0.005

    # 反对易 (Bob)
    constraints.append(d_dict["d_A1B0B2B0"] <= -d_dict["d_A1B2"] + tol_struct)
    # 反对易 (Alice)
    constraints.append(d_dict["d_A0A1A0B2"] <= -d_dict["d_A1B2"] + tol_struct)
    # 6体项一致性
    constraints.append(d_dict["d_A0A1A0B0B2B0"] >= d_dict["d_A1B2"] - tol_struct)

    # --- 6. 求解 ---
    prob = cp.Problem(cp.Minimize(func), constraints)
    try:
        prob.solve(solver=cp.MOSEK, verbose=False)
        if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            print(f"Minimize Result: {prob.value:.6f}")
            # print(f"Part 1: {p1.value:.4f}, Part 2: {p2.value:.4f}, Part 3: {p3.value:.4f}")
            return prob.value
        else:
            print(f"Solver Failed: {prob.status}")
            return None
    except Exception as e:
        print(f"Error: {e}")
        return None


# ==============================================================================
# 5. 主程序
# ==============================================================================
if __name__ == '__main__':
    # 1. 设置参数
    desc = np.array([2, 2, 2, 3])
    theta = np.pi / 6

    sin_2theta = np.sin(2 * theta)
    cos_2theta = np.cos(2 * theta)
    alpha_val = 2 * cos_2theta / np.sqrt(1 + sin_2theta ** 2)
    mu = np.arctan(sin_2theta)

    # 2. 旋转系数矩阵
    cos_mu = np.cos(mu)
    cos_2mu = np.cos(2 * mu)
    vec_rotated = [alpha_val, 0, 0, 0, 2 * cos_mu, 0, cos_2mu / cos_mu, -1.0 / cos_mu]
    coeff_ineq2 = convert_vector_to_matrix(vec_rotated)

    # 3. 计算理论界限 (标准 Tilted-CHSH 界限)
    # 注意: 旋转后的系数虽然能算出更高的经典界(3.78)，但那是基于对易关系的。
    # 我们的反对易约束锁定了量子区域，所以有效范围依然是标准界限。
    q_bound = np.sqrt(8 + 2 * alpha_val ** 2)
    c_bound = 2 + alpha_val

    print("-" * 50)
    print(f"Quantum Bound (Target Max): {q_bound:.6f}")
    print(f"Classical Bound (Target Min): {c_bound:.6f}")
    print("-" * 50)

    # 4. Case 1: Target = Max Quantum Violation
    print(">>> Running Case 1: Target = Q_Bound")
    fid_q = calculate_fidelity_at_target(q_bound, q_bound, c_bound, coeff_ineq2, desc, theta)

    # 5. Case 2: Target = Classical Bound
    print("\n>>> Running Case 2: Target = C_Bound")
    # 稍微加一点点 epsilon 避免边界数值问题
    fid_c = calculate_fidelity_at_target(c_bound + 0.0001, q_bound, c_bound, coeff_ineq2, desc, theta)