import numpy as np
import cvxpy as cp
import json
import os
import sys
import matplotlib.pyplot as plt

# ==============================================================================
# [导入依赖]
# ==============================================================================
try:
    from NPAHierarchy_Custom import npa_constraints
    from locate_gamma_2 import get_gamma_element
    import split_projector
    # [Task 1] 导入新接口
    from BellInequalityMax_new import compute_bell_limits
    # [Task 2] 导入 normal_vector_tilted_chsh
    from normal_vector_tilted_chsh import get_normalized_bell_coeffs
except ImportError as e:
    print(f"[Error] 导入依赖失败: {e}")
    sys.exit(1)


# ==============================================================================
# 1. 辅助函数
# ==============================================================================
def str_to_indices(basis_str: str, ma: int) -> list:
    """将算符字符串转换为索引列表"""
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
    # 解包顺序: A0, A1, B0, B1, A0B0, A0B1, A1B0, A1B1
    A0, A1, B0, B1, A0B0, A0B1, A1B0, A1B1 = vec
    M = np.zeros((3, 3))
    # Col 0: Marginals Alice
    M[1, 0] = A0
    M[2, 0] = A1
    # Row 0: Marginals Bob
    M[0, 1] = B0
    M[0, 2] = B1
    # Matrix body: Correlations
    M[1, 1] = A0B0
    M[1, 2] = A0B1
    M[2, 1] = A1B0
    M[2, 2] = A1B1
    return M


def build_fidelity_objective_from_json(G, ind_catalog, m_vec, theta_val, mu_val):
    """构建 Fidelity 目标函数"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(current_dir, 'func', 'fidelity_coeffs.json')

    if not os.path.exists(json_path):
        print(f"[Error] 找不到文件: {json_path}")
        return None

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    eval_context = {'np': np, 'theta': theta_val, 'mu': mu_val}
    objective_expr = 0

    for term_key, coeff_str in data.items():
        try:
            coeff_val = eval(coeff_str, {}, eval_context)
        except Exception:
            continue

        left_str, right_str = split_projector.split_projector_term(term_key)
        ma = m_vec[0]
        alice_seq = [] if left_str == "I" else str_to_indices(left_str, ma)
        bob_seq = [] if right_str == "I" else str_to_indices(right_str, ma)

        gamma_elem = get_gamma_element(G, ind_catalog, m_vec, alice_seq, bob_seq)
        if gamma_elem is None:
            raise ValueError(f"基底不足: 缺少项 {term_key}")

        objective_expr += coeff_val * gamma_elem

    return objective_expr


def build_tilted_chsh_constraint(G, ind_catalog, m_vec, coeff_matrix, target_param):
    """构建 Tilted-CHSH 约束"""
    expr = 0

    def get_E(al, bl):
        return get_gamma_element(G, ind_catalog, m_vec, al, bl)

    # Bob Marginals
    for y_local in range(2):
        c = coeff_matrix[0, y_local + 1]
        if abs(c) > 1e-9:
            term_Fy = get_E([], [m_vec[0] + y_local])
            expr += c * (2 * term_Fy - 1)

    # Alice Marginals
    for x_local in range(2):
        c = coeff_matrix[x_local + 1, 0]
        if abs(c) > 1e-9:
            term_Ex = get_E([x_local], [])
            expr += c * (2 * term_Ex - 1)

    # Correlations
    for x_local in range(2):
        for y_local in range(2):
            c = coeff_matrix[x_local + 1, y_local + 1]
            if abs(c) > 1e-9:
                term_ExFy = get_E([x_local], [m_vec[0] + y_local])
                term_Ex = get_E([x_local], [])
                term_Fy = get_E([], [m_vec[0] + y_local])
                # Correlator <AxBy> = 4<E x F y> - 2<E x> - 2<F y> + 1
                term_AxBy = 4 * term_ExFy - 2 * term_Ex - 2 * term_Fy + 1
                expr += c * term_AxBy

    return expr == target_param


def run_sdp_scan(vec_ineq, desc, theta, mu, num_points=30, label_prefix=""):
    """
    执行一次完整的 SDP 扫描过程
    Returns: (x_plot, y_plot) 用于绘图的数据
    """
    m_vec = desc[2:4]

    # 1. 转换矩阵并计算边界
    coeff_matrix = convert_vector_to_matrix(vec_ineq)
    print(f"\n[{label_prefix}] Calculating Bounds...")
    local_bound, quantum_bound = compute_bell_limits(coeff_matrix, desc, notation='fc', k=3)

    if local_bound is None or quantum_bound is None:
        print(f"[{label_prefix}] Error: Bounds calculation failed.")
        return [], []

    # [Task Requirement] 输出边界信息
    print(f"[{label_prefix}] Local Bound  : {local_bound:.6f}")
    print(f"[{label_prefix}] Quantum Bound: {quantum_bound:.6f}")

    # 这里的 linspace 保证了第一个点是 local_bound, 最后一个点是 quantum_bound
    target_vals = np.linspace(local_bound, quantum_bound, num_points)

    # 2. 构建 NPA 基础结构 (每次调用独立构建，避免参数混淆)
    custom_basis_str_list = [
        "", "E0", "E1", "F0", "F1",
        "E0 E1", "E1 E0", "F0 F1", "F1 F0",
        "E0 F0", "E0 F1", "E1 F0", "E1 F1",
        "E0 E1 E0", "F0 F1 F0", "F1 F0 F1",
        "E0 F0 F1", "E0 F1 F0"
    ]
    ma = m_vec[0]
    custom_basis_tuples = [
        tuple(str_to_indices(s, ma)) if s else () for s in custom_basis_str_list
    ]
    dummy_cg = np.zeros((3, 3))
    G, base_constraints, ind_catalog = npa_constraints(
        dummy_cg, desc, custom_basis=custom_basis_tuples, enforce_data=False
    )

    # 3. 构建参数和约束
    target_param = cp.Parameter()
    try:
        bell_constr = build_tilted_chsh_constraint(G, ind_catalog, m_vec, coeff_matrix, target_param)
        func = build_fidelity_objective_from_json(G, ind_catalog, m_vec, theta, mu)
        prob = cp.Problem(cp.Minimize(func), base_constraints + [bell_constr])
    except Exception as e:
        print(f"[{label_prefix}] Construct Error: {e}")
        return [], []

    # 4. 扫描
    results_fid = []
    results_v_norm = []

    denom = quantum_bound - local_bound

    print(f"[{label_prefix}] Scanning from Local to Quantum bound...")
    for idx, val in enumerate(target_vals):
        target_param.value = val

        # 归一化 V 计算
        if denom != 0:
            v_norm = (val - local_bound) / denom
        else:
            v_norm = 0.0

        try:
            if 'MOSEK' in cp.installed_solvers():
                prob.solve(solver=cp.MOSEK, verbose=False)
            else:
                prob.solve(solver=cp.SCS, verbose=False)
        except:
            if idx == 0 or idx == num_points - 1:
                print(f"  > 边界点求解失败 (Val={val:.4f})")
            continue

        # 记录结果
        fid_val = None
        if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            fid_val = prob.value
            results_fid.append(fid_val)
            results_v_norm.append(v_norm)

        # [Task Requirement] 输出边界处的 SDP 结果
        if idx == 0:
            print(f"  > [Local Bound]   Bell={val:.6f} | Fidelity={fid_val if fid_val else 'Failed'}")
        elif idx == num_points - 1:
            print(f"  > [Quantum Bound] Bell={val:.6f} | Fidelity={fid_val if fid_val else 'Failed'}")

    return results_v_norm, results_fid


def execute_comparison_and_plot(alpha_val, lambda_vals, vec_ineq, vec_ineq_2):
    """
    封装的执行函数：计算参数、运行扫描、绘图并保存。
    """
    print(f"[Exec] Starting comparison for Alpha={alpha_val}, Lambda={lambda_vals}")

    # 1. 派生物理参数
    x = (alpha_val ** 2) / 4.0
    sin_2theta = np.sqrt((1 - x) / (1 + x))
    theta = 0.5 * np.arcsin(sin_2theta)
    mu = np.arctan(sin_2theta)
    desc = np.array([2, 2, 2, 2])

    # 2. 运行扫描
    # Curve 1
    x1, y1 = run_sdp_scan(vec_ineq, desc, theta, mu, num_points=30, label_prefix="Standard")
    # Curve 2
    x2, y2 = run_sdp_scan(vec_ineq_2, desc, theta, mu, num_points=30, label_prefix="New")

    # 3. 绘图
    if not x1 or not x2:
        print("[Error] 缺少绘图数据")
        return

    plt.figure(figsize=(8, 6))

    plt.plot(x1, y1, 'o-', label=f'Standard Tilted-CHSH (alpha={alpha_val})', color='blue', markersize=4)
    plt.plot(x2, y2, 's--', label=f'New Inequality (lambda={lambda_vals})', color='red', markersize=4)

    plt.xlabel('Normalized Violation V = (Val - Local) / (Quantum - Local)')
    plt.ylabel('Minimum Fidelity')
    plt.title(f'Fidelity Robustness Comparison (Alpha={alpha_val})')
    plt.grid(True)
    plt.legend()

    # 标记边界
    plt.axvline(x=0.0, color='gray', linestyle=':', alpha=0.5)
    plt.axvline(x=1.0, color='gray', linestyle=':', alpha=0.5)

    # 保存
    current_dir = os.path.dirname(os.path.abspath(__file__))
    fig_dir = os.path.join(current_dir, 'figure')
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    save_path = os.path.join(fig_dir, 'tilted_alpha_figure3.png')
    plt.savefig(save_path)
    print(f"\n[Output] Comparison figure saved to: {save_path}")


# ==============================================================================
# 3. 主逻辑入口
# ==============================================================================
if __name__ == "__main__":
    print("=== Custom NPA SDP: Dual Curve Comparison ===")

    # 1. 设定输入参数
    alpha_val = 0.5
    lambda_vals = [0.212732, 0.287259, 0.287322, 0.212687]

    # 2. 准备两个不等式系数向量
    # Standard Tilted-CHSH
    vec_ineq = [alpha_val, 0, 0, 0, 1, 1, 1, -1]

    # New Inequality from Gradients
    try:
        vec_ineq_2 = get_normalized_bell_coeffs(alpha_val, lambda_vals)
        print(f"[Main] vec_ineq_2: {np.round(vec_ineq_2, 4)}")
    except Exception as e:
        print(f"[Error] Failed to generate vec_ineq_2: {e}")
        sys.exit(1)

    # 3. 执行比较绘图逻辑
    execute_comparison_and_plot(alpha_val, lambda_vals, vec_ineq, vec_ineq_2)