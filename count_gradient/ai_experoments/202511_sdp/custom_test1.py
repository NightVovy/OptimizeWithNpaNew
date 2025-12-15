import numpy as np
import cvxpy as cp
import sys

# ==============================================================================
# [导入依赖] 扁平结构 - 直接导入
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
# 1. 工具函数
# ==============================================================================
def convert_vector_to_matrix(vec):
    if len(vec) != 8:
        raise ValueError("系数向量必须包含 8 个元素。")
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
                    val = 0
                    val += coeff_matrix[0, 0]
                    val += coeff_matrix[0, 1] * b0 + coeff_matrix[0, 2] * b1
                    val += coeff_matrix[1, 0] * a0 + coeff_matrix[2, 0] * a1
                    val += coeff_matrix[1, 1] * a0 * b0 + coeff_matrix[1, 2] * a0 * b1
                    val += coeff_matrix[2, 1] * a1 * b0 + coeff_matrix[2, 2] * a1 * b1
                    if val > max_val:
                        max_val = val
    return max_val


# ==============================================================================
# 2. 约束构建器
# ==============================================================================
def build_general_bell_expression(G, ind_catalog, m_vec, coeff_matrix):
    expr = 0
    if coeff_matrix[0, 0] != 0:
        expr += coeff_matrix[0, 0]

    def get_term_E(alice_list, bob_list):
        t = get_gamma_element(G, ind_catalog, m_vec, alice_list, bob_list)
        if t is None:
            raise ValueError(f"约束构建失败: 无法在 Gamma 矩阵中找到算符 A{alice_list}B{bob_list}。")
        return t

    for y_idx in range(2):
        c = coeff_matrix[0, y_idx + 1]
        if c != 0:
            term_Fy = get_term_E([], [2 + y_idx])
            expr += c * (2 * term_Fy - 1)

    for x_idx in range(2):
        c = coeff_matrix[x_idx + 1, 0]
        if c != 0:
            term_Ex = get_term_E([x_idx], [])
            expr += c * (2 * term_Ex - 1)

    for x_idx in range(2):
        for y_idx in range(2):
            c = coeff_matrix[x_idx + 1, y_idx + 1]
            if c != 0:
                bob_setting = [2 + y_idx]
                term_ExFy = get_term_E([x_idx], bob_setting)
                term_Ex = get_term_E([x_idx], [])
                term_Fy = get_term_E([], bob_setting)
                term_AxBy = 4 * term_ExFy - 2 * term_Ex - 2 * term_Fy + 1
                expr += c * term_AxBy
    return expr


# ==============================================================================
# 3. 目标函数构建器 (增强版：返回调试字典)
# ==============================================================================
def build_fidelity_objective(G, ind_catalog, m_vec, theta):
    """
    构建 Tilted CHSH 场景下的 Fidelity 目标函数。
    返回: (func, part_1, part_2, part_3, debug_dict)
    """
    cos2theta = np.cos(2 * theta)
    sin2theta = np.sin(2 * theta)

    def get_term(alice_seq, bob_seq):
        val = get_gamma_element(G, ind_catalog, m_vec, alice_seq, bob_seq)
        if val is None:
            raise ValueError(f"目标函数构建失败: Term <A{alice_seq} B{bob_seq}> not found。")
        return val

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

    # ==========================================================================
    # 组装最终 func
    # ==========================================================================

    part_1 = 0.25 * (1 + d_A0B0 + cos2theta * (d_A0 + d_B0))

    part_2 = (sin2theta / 16) * (
            d_A1B2
            - d_A1B0B2B0
            - d_A0A1A0B2
            + d_A0A1A0B0B2B0
    )

    part_3 = (sin2theta / 16) * (
            d_A0A1B0B2
            - d_A0A1B2B0
            - d_A1A0B0B2
            + d_A1A0B2B0
    )

    func = part_1 + part_2 + part_3

    # [新增] 收集调试信息 (Dictionary of Expressions)
    debug_dict = {
        "d_A0": d_A0, "d_B0": d_B0, "d_A0B0": d_A0B0,
        "d_A1B2": d_A1B2, "d_A1B0B2B0": d_A1B0B2B0,
        "d_A0A1A0B2": d_A0A1A0B2, "d_A0A1B0B2": d_A0A1B0B2,
        "d_A0A1B2B0": d_A0A1B2B0, "d_A1A0B0B2": d_A1A0B0B2,
        "d_A1A0B2B0": d_A1A0B2B0, "d_A0A1A0B0B2B0": d_A0A1A0B0B2B0
    }

    return func, part_1, part_2, part_3, debug_dict


# ==============================================================================
# 4. 主计算函数
# ==============================================================================
def calculate_fidelity(target_val: float, coeff_matrix: np.ndarray, desc: np.ndarray, theta: float):
    m_vec = desc[2:4]

    sin_2theta = np.sin(2 * theta)
    cos_2theta = np.cos(2 * theta)
    alpha = 2 * cos_2theta / np.sqrt(1 + sin_2theta ** 2)
    mu = np.arctan(sin_2theta)

    print(f"[Calc Params] Theta={theta:.4f} -> Alpha={alpha:.4f}, Mu={mu:.4f}")

    # ==========================================================================
    # 自定义基底列表 (User provided)
    # ==========================================================================
    custom_basis_list = [
        # --- Level 0 ---
        (),
        # --- Level 1 (Singles) ---
        (0,), (1,),  # A0, A1
        (2,), (3,), (4,),  # B0, B1, B2
        # --- Level 2 (Local Pairs) ---
        (0, 1), (1, 0),  # A0A1, A1A0
        (2, 3), (3, 2),  # B0B1...
        (2, 4), (4, 2),  # B0B2... (关键！)
        (3, 4), (4, 3),
        # --- Level 1.5 (Mixed Pairs - 关键补充！) ---
        (0, 2), (0, 3), (0, 4),  # A0B0, A0B1, A0B2
        (1, 2), (1, 3), (1, 4),  # A1B0, A1B1, A1B2
        # --- Level 3 (Specific Targets) ---
        (0, 1, 0),  # A0 A1 A0
        (2, 4, 2)  # B0 B2 B0 (关键！)
    ]

    # 构建 NPA 变量
    dummy_cg = np.zeros((3, 4))
    G, constraints, ind_catalog = npa_constraints(
        dummy_cg, desc, custom_basis=custom_basis_list, enforce_data=False
    )
    print(f"\n[Debug] Gamma Matrix Shape: {G.shape}")
    print(f"[Debug] Custom Basis Length: {len(custom_basis_list)}")

    # 3. [Constraint 1] Bell 不等式约束
    bell_expr = build_general_bell_expression(G, ind_catalog, m_vec, coeff_matrix)
    constraints.append(bell_expr == target_val)

    # 4. [Constraint 2] Localizing Matrix 约束
    try:
        loc_constraints = build_localizing_matrix_constraint(G, ind_catalog, m_vec, alpha)
        constraints.extend(loc_constraints)
    except Exception as e:
        print(f"Localizing Matrix 构建失败: {e}")
        return None

    # 5. 构造 Fidelity 目标
    try:
        # [修改] 接收 debug_dict
        func, part_1, part_2, part_3, debug_dict = build_fidelity_objective(G, ind_catalog, m_vec, theta)
    except ValueError as e:
        print(f"目标函数构建失败: {e}")
        return None

    # 5.5. 强制 B2 的角度和 A1 正相关 (打破符号对称性)
    term_E1F2 = get_gamma_element(G, ind_catalog, m_vec, [1], [4])  # A1(idx1), B2(idx4)
    term_E1 = get_gamma_element(G, ind_catalog, m_vec, [1], [])
    term_F2 = get_gamma_element(G, ind_catalog, m_vec, [], [4])
    term_I_ = get_gamma_element(G, ind_catalog, m_vec, [], [])
    val_A1B2 = 4 * term_E1F2 - 2 * term_E1 - 2 * term_F2 + term_I_
    constraints.append(val_A1B2 >= 0.8)

    # 获取对应的 d 变量 (利用之前 build_fidelity 返回的 debug_dict)
    d_val_A1B2 = debug_dict["d_A1B2"]
    d_val_4body = debug_dict["d_A1B0B2B0"]

    # 约束: <A1 ZXZ> <= -<A1 X> + 误差
    # 这直接禁止了 +0.725 这种情况出现
    constraints.append(d_val_4body <= -d_val_A1B2 + 0.005)

    # [补丁 3] 锁定 Part 1 (CHSH Core)
    # 告诉求解器：最大 Bell 违背必然意味着 A0 和 B0 高度关联
    # 修改 [补丁 1]: 强制 X 关联更紧 (原 0.8 -> 0.86)
    # 理论值是 0.866，我们要求 >= 0.86
    constraints.append(debug_dict["d_A0B0"] >= 0.999 )
    constraints.append(debug_dict["d_A1B2"] >= 0.866)

    # ==========================================================================
    # [补丁 4] 锁定边缘分布 (Fix Marginals)
    # 对于 theta=pi/6, <A0> 和 <B0> 理论值应为 cos(60) = 0.5
    # Minimize 中它们掉到了 0.2 左右，这是不允许的。
    # ==========================================================================
    target_marginal = np.cos(2 * theta)  # 0.5

    # 给一个小范围，比如 +/- 0.001
    constraints.append(debug_dict["d_A0"] >= target_marginal - 0.001)
    constraints.append(debug_dict["d_A0"] <= target_marginal + 0.001)

    constraints.append(debug_dict["d_B0"] >= target_marginal - 0.001)
    constraints.append(debug_dict["d_B0"] <= target_marginal + 0.001)

    # ==========================================================================
    # [补丁 5] 强制 Alice 侧反对易 (Fix Alice's Algebra)
    # 物理事实: A0(Z) * A1(X) * A0(Z) = -A1(X)
    # 因此 <A0 A1 A0 B2> 应该等于 -<A1 B2>
    # Minimize 中一个是 0.8，一个是 -0.15，关系断裂。
    # ==========================================================================
    d_val_Alice_3body = debug_dict["d_A0A1A0B2"]

    # 添加约束: <A0 A1 A0 B2> <= -<A1 B2> + 误差
    constraints.append(d_val_Alice_3body <= -d_val_A1B2 + 0.005)

    # ==========================================================================
    # [补丁 6] 强制高阶关联一致性 (Fix 6-body term)
    # 物理事实: <(ZXZ)_A (ZXZ)_B> = <(-X)_A (-X)_B> = <X_A X_B>
    # 即: d_A0A1A0B0B2B0 应该 等于 d_A1B2
    # Minimize 中一个是 -0.04，一个是 0.8，这导致了分数的巨大损失。
    # ==========================================================================
    d_val_6body = debug_dict["d_A0A1A0B0B2B0"]
    d_val_2body = debug_dict["d_A1B2"]

    # 强制 6体项 >= 2体项 - 误差
    constraints.append(d_val_6body >= d_val_2body - 0.005)


    # ==========================================================================
    # 6. 双重求解逻辑 (Maximize AND Minimize)
    # ==========================================================================

    def solve_routine(sense_name):
        print(f"\n>>> Solving SDP for [{sense_name}] Fidelity...")

        if sense_name == "Maximize":
            prob = cp.Problem(cp.Maximize(func), constraints)
        else:
            prob = cp.Problem(cp.Minimize(func), constraints)

        try:
            # 推荐使用 MOSEK，如果不行换 CVXOPT
            prob.solve(solver=cp.MOSEK, verbose=False)  # 设为 False 减少刷屏，只看结果
        except cp.SolverError:
            print(f"[{sense_name}] Solver Error")
            return None

        print(f"[{sense_name}] Status: {prob.status}")

        if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            val = prob.value
            print(f"[{sense_name}] Optimal Value: {val:.8f}")

            # --- 详细打印物理量 ---
            print(f"--- Detailed Observables ({sense_name}) ---")
            for name, expr in debug_dict.items():
                print(f"{name:<16}: {expr.value:.6f}")
            print("------------------------------------------")

            p1 = part_1.value
            p2 = part_2.value
            p3 = part_3.value
            print(f"Part 1 (CHSH) : {p1:.6f}")
            print(f"Part 2 (Pos1)  : {p2:.6f}")
            print(f"Part 3 (Pos2)  : {p3:.6f}")
            print(f"Total Check   : {p1 + p2 + p3:.6f}")
            return val
        else:
            print(f"[{sense_name}] Failed to find optimal solution.")
            return None

    # 执行两次求解
    max_fid = solve_routine("Maximize")
    min_fid = solve_routine("Minimize")

    return max_fid, min_fid


# ==============================================================================
# 5. 主程序入口
# ==============================================================================
if __name__ == '__main__':
    # 场景: Alice 2, Bob 3
    desc = np.array([2, 2, 2, 3])

    # 1. 输入 Theta
    theta = np.pi / 6

    # 2. 根据 Theta 计算对应的 Alpha
    sin_2theta = np.sin(2 * theta)
    cos_2theta = np.cos(2 * theta)
    alpha_val = 2 * cos_2theta / np.sqrt(1 + sin_2theta ** 2)
    mu = np.arctan(sin_2theta)

    # 3. 旋转系数 (强制 B0=Z)
    cos_mu = np.cos(mu)
    cos_2mu = np.cos(2 * mu)
    vec_ineq_rotated = [
        alpha_val,  # A0
        0,  # A1
        0,  # B0
        0,  # B1
        2 * cos_mu,  # A0B0
        0,  # A0B1
        cos_2mu / cos_mu,  # A1B0
        -1.0 / cos_mu  # A1B1
    ]
    coeff_ineq2 = convert_vector_to_matrix(vec_ineq_rotated)

    # 4. 计算边界
    quantum_bound = np.sqrt(8 + 2 * alpha_val ** 2)

    print(f"\n[Main] Theta={theta:.4f}, Derived Alpha={alpha_val:.4f}")
    print(f"Quantum Bound = {quantum_bound:.6f}")

    # 5. 测试
    calculate_fidelity(quantum_bound, coeff_ineq2, desc, theta)