import numpy as np
import cvxpy as cp
import json
import os
import sys
import matplotlib.pyplot as plt  # 引入绘图库

# ==============================================================================
# [导入依赖]
# ==============================================================================
try:
    from NPAHierarchy_Custom import npa_constraints
    from locate_gamma_2 import get_gamma_element
    import npa_support
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
    Alice: E0->0, E1->1
    Bob:   F0->ma+0, F1->ma+1
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
            indices.append(ma + 0)
        elif t == "F1":
            indices.append(ma + 1)
        else:
            raise ValueError(f"未知的算符: {t}")
    return indices


def convert_vector_to_matrix(vec):
    """将由8个系数组成的向量转换为 3x3 矩阵"""
    if len(vec) != 8:
        raise ValueError("系数向量必须包含 8 个元素。")
    A0, A1, B0, B1, A0B0, A0B1, A1B0, A1B1 = vec
    M = np.zeros((3, 3))
    M[0, 1] = B0
    M[0, 2] = B1
    M[1, 0] = A0
    M[2, 0] = A1
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
    读取 json 文件，解析系数和算符，构建目标函数。
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(current_dir, 'func', 'fidelity_coeffs.json')

    if not os.path.exists(json_path):
        print(f"[Error] 找不到文件: {json_path}")
        return None, {}

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    eval_context = {'np': np, 'theta': theta_val, 'mu': mu_val}
    objective_expr = 0
    debug_terms = {}

    print(f"\n[Info] 构建 Fidelity 目标函数 (共 {len(data)} 项)...")

    for term_key, coeff_str in data.items():
        try:
            coeff_val = eval(coeff_str, {}, eval_context)
        except Exception as e:
            print(f"[Warning] 无法计算系数 '{coeff_str}': {e}")
            continue

        left_str, right_str = split_projector.split_projector_term(term_key)
        ma = m_vec[0]
        alice_seq = [] if left_str == "I" else str_to_indices(left_str, ma)
        bob_seq = [] if right_str == "I" else str_to_indices(right_str, ma)

        gamma_elem = get_gamma_element(G, ind_catalog, m_vec, alice_seq, bob_seq)
        if gamma_elem is None:
            raise ValueError(f"基底不足: 缺少项 {term_key}")

        objective_expr += coeff_val * gamma_elem
        debug_terms[term_key] = (coeff_val, gamma_elem)

    return objective_expr, debug_terms


# ==============================================================================
# 3. 约束构建器 (Tilted CHSH Expectation)
# ==============================================================================
def build_tilted_chsh_constraint(G, ind_catalog, m_vec, coeff_matrix, target_param):
    """
    构建 Tilted-CHSH 表达式 == target_param 的约束。
    target_param 可以是 float 或 cvxpy.Parameter
    """
    expr = 0

    def get_E(alice_list, bob_list):
        el = get_gamma_element(G, ind_catalog, m_vec, alice_list, bob_list)
        if el is None:
            raise ValueError(f"Tilted-CHSH 约束构建失败: 缺少项 A{alice_list} B{bob_list}")
        return el

    # Bob Marginals
    for y_local in range(2):
        c = coeff_matrix[0, y_local + 1]
        if c != 0:
            y_global = m_vec[0] + y_local
            term_Fy = get_E([], [y_global])
            expr += c * (2 * term_Fy - 1)

    # Alice Marginals
    for x_local in range(2):
        c = coeff_matrix[x_local + 1, 0]
        if c != 0:
            term_Ex = get_E([x_local], [])
            expr += c * (2 * term_Ex - 1)

    # Correlations
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

    return expr == target_param


# ==============================================================================
# 4. 主逻辑
# ==============================================================================
def main():
    print("=== Custom NPA SDP: Scanning Tilted-CHSH Values ===")

    # 1. 物理参数设定
    # ==========================
    alpha_val = 0.5

    # 派生 theta, mu
    x = (alpha_val ** 2) / 4.0
    sin_2theta = np.sqrt((1 - x) / (1 + x))
    theta = 0.5 * np.arcsin(sin_2theta)
    mu = np.arctan(sin_2theta)

    desc = np.array([2, 2, 2, 2])
    m_vec = desc[2:4]

    print(f"[Params] Alpha={alpha_val}, Theta={theta:.4f}, Mu={mu:.4f}")

    # 定义扫描范围
    # Local Bound = 2 + alpha
    local_bound = 2.0 + alpha_val
    # Quantum Bound = sqrt(8 + 2*alpha^2)
    quantum_bound = np.sqrt(8 + 2 * alpha_val ** 2)

    num_points = 30
    target_vals = np.linspace(local_bound, quantum_bound, num_points)

    print(f"[Range] Local Bound  = {local_bound:.6f}")
    print(f"[Range] Quantum Bound = {quantum_bound:.6f}")
    print(f"[Range] Generating {num_points} points for scan.")

    # Tilted-CHSH 系数
    vec_ineq_rotated = [alpha_val, 0, 0, 0, 1, 1, 1, -1]
    coeff_matrix = convert_vector_to_matrix(vec_ineq_rotated)

    # 2. 构建 NPA 问题 (只构建一次)
    # ==========================
    custom_basis_str_list = [
        "", "E0", "E1", "F0", "F1",
        "E0 E1", "E1 E0", "F0 F1", "F1 F0",
        "E0 F0", "E0 F1", "E1 F0", "E1 F1",
        "E0 E1 E0", "F0 F1 F0", "F1 F0 F1",
        "E0 F0 F1", "E0 F1 F0"
    ]
    ma = m_vec[0]
    custom_basis_tuples = [
        tuple(str_to_indices(s, ma)) if s else ()
        for s in custom_basis_str_list
    ]

    dummy_cg = np.zeros((3, 3))
    print("\n[NPA] 构建 Gamma 矩阵结构...")
    G, base_constraints, ind_catalog = npa_constraints(
        dummy_cg, desc,
        custom_basis=custom_basis_tuples,
        enforce_data=False
    )

    # 3. 构建参数化问题
    # ==========================
    # 定义 Parameter 以便在循环中更新，避免重复构建矩阵
    target_param = cp.Parameter()

    # 约束: Tilted-CHSH == target_param
    try:
        bell_constr = build_tilted_chsh_constraint(G, ind_catalog, m_vec, coeff_matrix, target_param)
        # 将新约束加入基础约束列表
        all_constraints = base_constraints + [bell_constr]
    except ValueError as e:
        print(f"[Error] 约束构建失败: {e}")
        return

    # 目标函数
    try:
        func, _ = build_fidelity_objective_from_json(G, ind_catalog, m_vec, theta, mu)
    except ValueError as e:
        print(f"[Error] 目标函数构建失败: {e}")
        return

    # 编译 SDP 问题
    prob = cp.Problem(cp.Minimize(func), all_constraints)

    # 4. 循环求解
    # ==========================
    results_fid = []
    results_val = []

    print("\n>>> 开始扫描计算...")
    for idx, val in enumerate(target_vals):
        target_param.value = val

        try:
            # 使用 MOSEK 或 SCS
            if 'MOSEK' in cp.installed_solvers():
                prob.solve(solver=cp.MOSEK, verbose=False)
            else:
                prob.solve(solver=cp.SCS, verbose=False)
        except cp.SolverError:
            print(f"  [{idx + 1}/{num_points}] Solver Error at val={val:.4f}")
            results_fid.append(None)
            results_val.append(val)
            continue

        if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            fid = prob.value
            results_fid.append(fid)
            results_val.append(val)
            print(f"  [{idx + 1}/{num_points}] Val={val:.4f} -> Fidelity={fid:.6f}")
        else:
            print(f"  [{idx + 1}/{num_points}] Val={val:.4f} -> Status: {prob.status}")
            results_fid.append(None)
            results_val.append(val)

    # 5. 绘图与保存 (修改部分)
    # ==========================
    # 过滤掉 None 值
    valid_data = [(v, f) for v, f in zip(results_val, results_fid) if f is not None]
    if not valid_data:
        print("[Error] 没有有效的计算结果，无法绘图。")
        return

    x_plot, y_plot = zip(*valid_data)

    # 创建绘图对象
    plt.figure(figsize=(8, 6))
    plt.plot(x_plot, y_plot, 'o-', label=f'Alpha={alpha_val}')
    plt.xlabel('Tilted-CHSH Value')
    plt.ylabel('Minimum Fidelity')
    plt.title(f'Fidelity vs Tilted-CHSH Violation (Alpha={alpha_val})')
    plt.grid(True)
    plt.legend()

    # 标记边界
    plt.axvline(x=local_bound, color='r', linestyle='--', alpha=0.5, label='Local Bound')
    plt.axvline(x=quantum_bound, color='g', linestyle='--', alpha=0.5, label='Quantum Bound')

    # --- 文件夹检查与保存逻辑 ---
    current_dir = os.path.dirname(os.path.abspath(__file__))
    fig_dir = os.path.join(current_dir, 'figure')

    # 检查是否存在，不存在则创建
    if not os.path.exists(fig_dir):
        print(f"[Info] 文件夹 '{fig_dir}' 不存在，正在创建...")
        os.makedirs(fig_dir)
    else:
        print(f"[Info] 文件夹 '{fig_dir}' 已存在。")

    # 保存图片
    save_path = os.path.join(fig_dir, 'tilted_alpha_figure1.png')
    plt.savefig(save_path)
    print(f"\n[Output] 图片已保存至: {save_path}")


if __name__ == "__main__":
    main()