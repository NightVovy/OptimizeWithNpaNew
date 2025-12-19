import numpy as np
import cvxpy as cp
import json
import os
import sys

# ==============================================================================
# [导入依赖]
# ==============================================================================
# 尝试导入同级目录下的模块
try:
    from NPAHierarchy_Custom import npa_constraints
    from locate_gamma_2 import get_gamma_element
    import npa_support

    # 尝试导入 split_projector
    import split_projector
except ImportError as e:
    print(f"[Error] 导入依赖失败: {e}")
    print("请确保 NPAHierarchy_Custom.py, locate_gamma_2.py, npa_support.py, split_projector.py 在正确位置。")
    sys.exit(1)


# ==============================================================================
# 1. 辅助函数：算符字符串转索引
# ==============================================================================
def str_to_indices(basis_str: str, ma: int) -> list:
    """
    将算符字符串（如 "E0 E1"）转换为索引列表（如 [0, 1]）。
    映射规则:
      Alice: E0->0, E1->1
      Bob:   F0->ma+0, F1->ma+1 (这里ma=2，所以 F0->2, F1->3)
    """
    if not basis_str.strip():
        return []

    tokens = basis_str.split()
    indices = []
    for t in tokens:
        if t == "E0":
            indices.append(0)
        elif t == "E1":
            indices.append(1)
        elif t == "F0":
            indices.append(ma + 0)  # 2
        elif t == "F1":
            indices.append(ma + 1)  # 3
        else:
            raise ValueError(f"未知的算符: {t}")
    return indices


def convert_vector_to_matrix(vec):
    """将由8个系数组成的向量转换为 3x3 矩阵 (对应 I, A0, A1 和 I, B0, B1)"""
    # 向量顺序: A0, A1, B0, B1, A0B0, A0B1, A1B0, A1B1
    if len(vec) != 8:
        raise ValueError("系数向量必须包含 8 个元素。")
    A0, A1, B0, B1, A0B0, A0B1, A1B0, A1B1 = vec

    # Matrix rows: I, A0, A1
    # Matrix cols: I, B0, B1 (Bob index 0->B0, 1->B1 for matrix constr)
    M = np.zeros((3, 3))

    # Row 0: Marginals of Bob (coeffs for I*B0, I*B1)
    M[0, 1] = B0
    M[0, 2] = B1

    # Col 0: Marginals of Alice (coeffs for A0*I, A1*I)
    M[1, 0] = A0
    M[2, 0] = A1

    # Interactions
    M[1, 1] = A0B0
    M[1, 2] = A0B1
    M[2, 1] = A1B0
    M[2, 2] = A1B1

    return M


# ==============================================================================
# 2. 目标函数构建器 (Fidelity from JSON)
# ==============================================================================
def build_fidelity_objective_from_json(G, ind_catalog, m_vec, theta_val, mu_val):
    """
    读取 json 文件，解析系数和算符，并在 Gamma 矩阵中找到对应元素，构建目标函数。
    """
    # 1. 路径定位
    current_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(current_dir, 'func', 'fidelity_coeffs.json')

    if not os.path.exists(json_path):
        print(f"[Error] 找不到文件: {json_path}")
        return None, {}

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 2. 准备 eval 上下文
    eval_context = {
        'np': np,
        'theta': theta_val,
        'mu': mu_val
    }

    objective_expr = 0
    debug_terms = {}  # 存储每一项的值以便调试

    print(f"\n[Info] 开始构建 Fidelity 目标函数 (共 {len(data)} 项)...")

    for term_key, coeff_str in data.items():
        # A. 计算数值系数
        try:
            coeff_val = eval(coeff_str, {}, eval_context)
        except Exception as e:
            print(f"[Warning] 无法计算系数 '{coeff_str}': {e}")
            continue

        # B. 拆分算符 (Alice Left, Bob Right)
        left_str, right_str = split_projector.split_projector_term(term_key)

        # C. 转换为索引
        ma = m_vec[0]
        alice_seq = [] if left_str == "I" else str_to_indices(left_str, ma)
        bob_seq = [] if right_str == "I" else str_to_indices(right_str, ma)

        # D. 在 Gamma 矩阵中查找
        gamma_elem = get_gamma_element(G, ind_catalog, m_vec, alice_seq, bob_seq)

        if gamma_elem is None:
            print(f"[Error] Gamma 矩阵中未找到算符项: <A:{alice_seq}, B:{bob_seq}> (原始: {term_key})")
            raise ValueError(f"基底不足: 缺少项 {term_key}")

        # E. 累加
        objective_expr += coeff_val * gamma_elem

        # 记录调试信息
        debug_terms[term_key] = (coeff_val, gamma_elem)

    return objective_expr, debug_terms


# ==============================================================================
# 3. 约束构建器 (Tilted CHSH Expectation)
# ==============================================================================
def build_tilted_chsh_constraint(G, ind_catalog, m_vec, coeff_matrix, target_val):
    """
    构建 Tilted-CHSH 表达式 == target_val 的约束。
    基于 A_x = 2*E_x - I, B_y = 2*F_y - I 的转换关系。
    """
    expr = 0

    # 定义查找辅助函数
    def get_E(alice_list, bob_list):
        el = get_gamma_element(G, ind_catalog, m_vec, alice_list, bob_list)
        if el is None:
            raise ValueError(f"Tilted-CHSH 约束构建失败: 缺少项 A{alice_list} B{bob_list}")
        return el

    # 2. Bob 的单体项 (Marginals) -> B_y = 2 F_y - I
    for y_local in range(2):
        c = coeff_matrix[0, y_local + 1]  # col 1, 2
        if c != 0:
            y_global = m_vec[0] + y_local
            term_Fy = get_E([], [y_global])
            expr += c * (2 * term_Fy - 1)

    # 3. Alice 的单体项 (Marginals) -> A_x = 2 E_x - I
    for x_local in range(2):
        c = coeff_matrix[x_local + 1, 0]  # row 1, 2
        if c != 0:
            term_Ex = get_E([x_local], [])
            expr += c * (2 * term_Ex - 1)

    # 4. 关联项 (Correlations) -> A_x B_y = (2 Ex - I)(2 Fy - I)
    for x_local in range(2):
        for y_local in range(2):
            c = coeff_matrix[x_local + 1, y_local + 1]
            if c != 0:
                y_global = m_vec[0] + y_local
                term_ExFy = get_E([x_local], [y_global])
                term_Ex = get_E([x_local], [])
                term_Fy = get_E([], [y_global])

                term_AxBy = 4 * term_ExFy - 2 * term_Ex - 2 * term_Fy + 1
                expr += c * term_AxBy

    return expr == target_val


# ==============================================================================
# 4. 主逻辑
# ==============================================================================
def main():
    print("=== Custom NPA SDP for Fidelity with Tilted-CHSH (SWAP Method) [Mode: Alpha Input] ===")

    # 1. 参数设定 (核心修改部分)
    # ==========================
    # 设定 alpha
    alpha_val = 0.5

    # 根据公式计算 sin(2theta)
    # sin(2theta) = sqrt( (1 - a^2/4) / (1 + a^2/4) )
    x = (alpha_val ** 2) / 4.0
    sin_2theta = np.sqrt((1 - x) / (1 + x))

    # 计算 theta: theta = 0.5 * arcsin(sin(2theta))
    theta = 0.5 * np.arcsin(sin_2theta)

    # 计算 mu: tan(mu) = sin(2theta) -> mu = arctan(sin(2theta))
    mu = np.arctan(sin_2theta)

    # 描述符
    desc = np.array([2, 2, 2, 2])  # Alice 2 in 2 out, Bob 2 in 2 out
    m_vec = desc[2:4]  # [2, 2]

    print(f"[Params] Alpha (Input) = {alpha_val:.6f}")
    print(f"[Params] Theta (Derived) = {theta:.6f} (rad)")
    print(f"[Params] Mu    (Derived) = {mu:.6f} (rad)")

    # Target Value (Quantum Bound)
    quantum_bound = np.sqrt(8 + 2 * alpha_val ** 2)
    local_bound = 2 + alpha_val
    target_val = local_bound
    print(f"[Target] Tilted-CHSH Value = {target_val:.8f}")

    # Tilted-CHSH 系数向量
    # A0, A1, B0, B1, A0B0, A0B1, A1B0, A1B1
    vec_ineq_rotated = [
        alpha_val,  # A0
        0,  # A1
        0,  # B0
        0,  # B1
        1,  # A0B0
        1,  # A0B1
        1,  # A1B0
        -1  # A1B1
    ]
    coeff_matrix = convert_vector_to_matrix(vec_ineq_rotated)

    # 2. 构建 NPA Hierarchy (保持不变)
    # ==========================
    # 自定义基底 (字符串形式，需转 tuple)
    custom_basis_str_list = [
        "",  # I
        "E0", "E1", "F0", "F1",
        "E0 E1", "E1 E0", "F0 F1", "F1 F0",
        "E0 F0", "E0 F1", "E1 F0", "E1 F1",
        "E0 E1 E0", "F0 F1 F0", "F1 F0 F1",
        "E0 F0 F1", "E0 F1 F0"
    ]

    # 转换为 tuple (索引)
    custom_basis_tuples = []
    ma = m_vec[0]
    for s in custom_basis_str_list:
        if s == "":
            custom_basis_tuples.append(())
        else:
            custom_basis_tuples.append(tuple(str_to_indices(s, ma)))

    # 调用 NPA 构建器
    dummy_cg = np.zeros((3, 3))

    print("\n[NPA] 构建 Gamma 矩阵...")
    G, constraints, ind_catalog = npa_constraints(
        dummy_cg, desc,
        custom_basis=custom_basis_tuples,
        enforce_data=False
    )
    print(f"[NPA] Gamma 维度: {G.shape}")

    # 3. 添加约束 (保持不变)
    # ==========================
    # 约束 1: Tilted-CHSH == target_val
    try:
        bell_constr = build_tilted_chsh_constraint(G, ind_catalog, m_vec, coeff_matrix, target_val)
        constraints.append(bell_constr)
        print("[Constraint] Tilted-CHSH 约束已添加")
    except ValueError as e:
        print(f"[Error] Tilted-CHSH 约束构建失败: {e}")
        return

    # 4. 构建目标函数 (保持不变)
    # ==========================
    try:
        func, debug_terms_map = build_fidelity_objective_from_json(G, ind_catalog, m_vec, theta, mu)
    except ValueError as e:
        print(f"[Error] 目标函数构建失败: {e}")
        return

    # 5. 求解 SDP (保持不变)
    # ==========================
    print("\n>>> 开始求解 SDP (Minimize Fidelity)...")
    prob = cp.Problem(cp.Minimize(func), constraints)

    # 尝试求解
    try:
        # 优先使用 MOSEK，如果不可用则使用 SCS 或 CVXOPT
        if 'MOSEK' in cp.installed_solvers():
            prob.solve(solver=cp.MOSEK, verbose=False)
        else:
            prob.solve(solver=cp.SCS, verbose=False)

    except cp.SolverError as e:
        print(f"[Solver Error] {e}")
        return

    print(f"Status: {prob.status}")

    if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        min_fid = prob.value
        print(f"\n[Result] SDP Minimum Fidelity: {min_fid:.8f}")

        # 打印详细结果
        print("\n--- Detailed Terms (Projector -> Value) ---")
        sorted_keys = sorted(debug_terms_map.keys())
        total_check = 0.0

        for k in sorted_keys:
            coeff, expr = debug_terms_map[k]
            # 获取表达式的值
            if isinstance(expr, cp.Expression) or isinstance(expr, cp.Variable):
                val = expr.value
            else:
                val = expr  # 可能是常数

            term_contrib = coeff * val
            total_check += term_contrib

            print(f"{k:<20} | C: {coeff:8.4f} | E: {val:8.4f} | -> {term_contrib:8.4f}")

        print("-" * 60)
        print(f"Sum of terms: {total_check:.8f}")
        print(f"Target Value (Used in Constraint): {target_val:.8f}")

    else:
        print("[Result] 未找到最优解。")


if __name__ == "__main__":
    main()