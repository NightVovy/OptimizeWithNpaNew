import numpy as np
import cvxpy as cp
import sys
import os
import matplotlib.pyplot as plt

# ==============================================================================
# [设置路径]
# ==============================================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# ==============================================================================
# [导入依赖]
# ==============================================================================
try:
    from NPAHierarchy_3 import npa_constraints
    from locate_gamma_2 import get_gamma_element
    # [Task 1] 导入新接口
    from BellInequalityMax_new import compute_bell_limits
except ImportError as e:
    print(f"导入错误: {e}")
    print(f"请检查 NPAHierarchy_3.py, BellInequalityMax_new.py 等文件是否存在于父目录: {parent_dir}")
    sys.exit(1)


# ==============================================================================
# 1. 工具函数：系数转换与经典界计算
# ==============================================================================
def convert_vector_to_matrix(vec):
    if len(vec) != 8:
        raise ValueError("系数向量必须包含 8 个元素。")
    A0, A1, B0, B1, A0B0, A0B1, A1B0, A1B1 = vec
    M = np.zeros((3, 3))
    M[0, 1], M[0, 2] = B0, B1
    M[1, 0], M[1, 1], M[1, 2] = A0, A0B0, A0B1
    M[2, 0], M[2, 1], M[2, 2] = A1, A1B0, A1B1
    return M


# (此函数已不再使用，但为了保持代码结构完整性保留，或可直接忽略)
def calculate_classical_bound(coeff_matrix):
    max_val = -np.inf
    for a0 in [-1, 1]:
        for a1 in [-1, 1]:
            for b0 in [-1, 1]:
                for b1 in [-1, 1]:
                    val = coeff_matrix[0, 0]  # K
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
            raise ValueError(f"无法在 Gamma 矩阵中找到算符 A{alice_list}B{bob_list}。")
        return t

    # Bob Singles
    for y_idx in range(2):
        c = coeff_matrix[0, y_idx + 1]
        if c != 0:
            term_Fy = get_term_E([], [2 + y_idx])
            expr += c * (2 * term_Fy - 1)

    # Alice Singles
    for x_idx in range(2):
        c = coeff_matrix[x_idx + 1, 0]
        if c != 0:
            term_Ex = get_term_E([x_idx], [])
            expr += c * (2 * term_Ex - 1)

    # Correlations
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
# 3. Fidelity 目标函数构建器
# ==============================================================================
def build_fidelity_objective(G, ind_catalog, m_vec):
    def get_term(alice_seq, bob_seq):
        val = get_gamma_element(G, ind_catalog, m_vec, alice_seq, bob_seq)
        if val is None:
            raise ValueError(f"Term <A{alice_seq} B{bob_seq}> not found.")
        return val

    # --- 基础定义 ---
    A0, A1 = 0, 1
    B0, B1 = 2, 3
    I = []

    # --- (A) 纯单边项 (Pure Local Terms) ---
    term_I = get_term(I, I)

    # Alice Singles & Strings
    term_E0 = get_term([A0], I)
    term_E1 = get_term([A1], I)
    term_E0E1 = get_term([A0, A1], I)
    term_E1E0 = get_term([A1, A0], I)
    term_E0E1E0 = get_term([A0, A1, A0], I)  # Length 3

    # Bob Singles & Strings
    term_F0 = get_term(I, [B0])
    term_F1 = get_term(I, [B1])
    term_F0F1 = get_term(I, [B0, B1])
    term_F1F0 = get_term(I, [B1, B0])
    term_F0F1F0 = get_term(I, [B0, B1, B0])  # Length 3

    # --- (B) 1-1 交叉项 (Cross Terms 1-1) ---
    term_E0_F0 = get_term([A0], [B0])
    term_E0_F1 = get_term([A0], [B1])
    term_E1_F0 = get_term([A1], [B0])
    term_E1_F1 = get_term([A1], [B1])

    # --- (C) 2-1 和 1-2 交叉项 (Cross Terms 2-1, 1-2) ---
    # Alice 2, Bob 1
    term_E0E1_F0 = get_term([A0, A1], [B0])
    term_E0E1_F1 = get_term([A0, A1], [B1])
    term_E1E0_F0 = get_term([A1, A0], [B0])
    term_E1E0_F1 = get_term([A1, A0], [B1])

    # Alice 1, Bob 2
    term_E0_F0F1 = get_term([A0], [B0, B1])
    term_E0_F1F0 = get_term([A0], [B1, B0])
    term_E1_F0F1 = get_term([A1], [B0, B1])
    term_E1_F1F0 = get_term([A1], [B1, B0])

    # --- (D) 3-1 和 1-3 交叉项 (Cross Terms 3-1, 1-3) ---
    # Alice 3, Bob 1
    term_E0E1E0_F0 = get_term([A0, A1, A0], [B0])
    term_E0E1E0_F1 = get_term([A0, A1, A0], [B1])

    # Alice 1, Bob 3
    term_E0_F0F1F0 = get_term([A0], [B0, B1, B0])
    term_E1_F0F1F0 = get_term([A1], [B0, B1, B0])

    # --- (E) 2-2 交叉项 (Cross Terms 2-2) ---
    term_E0E1_F0F1 = get_term([A0, A1], [B0, B1])
    term_E0E1_F1F0 = get_term([A0, A1], [B1, B0])
    term_E1E0_F0F1 = get_term([A1, A0], [B0, B1])
    term_E1E0_F1F0 = get_term([A1, A0], [B1, B0])

    # --- (F) 3-2 和 2-3 交叉项 (用于 6-body 展开) ---
    term_E0E1E0_F0F1 = get_term([A0, A1, A0], [B0, B1])
    term_E0E1E0_F1F0 = get_term([A0, A1, A0], [B1, B0])

    term_E0E1_F0F1F0 = get_term([A0, A1], [B0, B1, B0])
    term_E1E0_F0F1F0 = get_term([A1, A0], [B0, B1, B0])

    # --- (G) 3-3 交叉项 (最高阶) ---
    term_E0E1E0_F0F1F0 = get_term([A0, A1, A0], [B0, B1, B0])

    # ==============================================================================
    # 6. 组装目标函数各部分 (d_Terms)
    # ==============================================================================

    # --- 2-body terms ---
    d_A0B0 = 4 * term_E0_F0 - 2 * term_E0 - 2 * term_F0 + term_I
    d_A1B1 = 4 * term_E1_F1 - 2 * term_E1 - 2 * term_F1 + term_I
    d_A0B1 = 4 * term_E0_F1 - 2 * term_E0 - 2 * term_F1 + term_I
    d_A1B0 = 4 * term_E1_F0 - 2 * term_E1 - 2 * term_F0 + term_I

    # 显式计算 Standard CHSH Value
    d_CHSH_std = d_A0B0 + d_A0B1 + d_A1B0 - d_A1B1

    # --- 4-body terms (Mixed 3-1 split) ---
    d_A0A1A0B1 = (16 * term_E0E1E0_F1
                  - 8 * (term_E0E1_F1 + term_E1E0_F1 + term_E0E1E0)
                  + 4 * (term_E0E1 + term_E1E0 + term_E1_F1)
                  - 2 * (term_E1 + term_F1)
                  + term_I)

    d_A0A1A0B0 = (16 * term_E0E1E0_F0
                  - 8 * (term_E0E1_F0 + term_E1E0_F0 + term_E0E1E0)
                  + 4 * (term_E0E1 + term_E1E0 + term_E1_F0)
                  - 2 * (term_E1 + term_F0)
                  + term_I)

    d_A1B0B1B0 = (16 * term_E1_F0F1F0
                  - 8 * (term_F0F1F0 + term_E1_F0F1 + term_E1_F1F0)
                  + 4 * (term_E1_F1 + term_F0F1 + term_F1F0)
                  - 2 * (term_E1 + term_F1)
                  + term_I)

    d_A0B0B1B0 = (16 * term_E0_F0F1F0
                  - 8 * (term_F0F1F0 + term_E0_F0F1 + term_E0_F1F0)
                  + 4 * (term_E0_F1 + term_F0F1 + term_F1F0)
                  - 2 * (term_E0 + term_F1)
                  + term_I)

    # --- 4-body terms (Mixed 2-2 split) ---
    d_A0A1B0B1 = (16 * term_E0E1_F0F1
                  - 8 * (term_E0E1_F0 + term_E0E1_F1 + term_E0_F0F1 + term_E1_F0F1)
                  + 4 * (term_E0_F0 + term_E0_F1 + term_E1_F0 + term_E1_F1)
                  + 4 * (term_E0E1 + term_F0F1)
                  - 2 * (term_E0 + term_E1 + term_F0 + term_F1)
                  + term_I)

    d_A0A1B1B0 = (16 * term_E0E1_F1F0
                  - 8 * (term_E0E1_F1 + term_E0E1_F0 + term_E0_F1F0 + term_E1_F1F0)
                  + 4 * (term_E0_F1 + term_E0_F0 + term_E1_F1 + term_E1_F0)
                  + 4 * (term_E0E1 + term_F1F0)
                  - 2 * (term_E0 + term_E1 + term_F1 + term_F0)
                  + term_I)

    d_A1A0B0B1 = (16 * term_E1E0_F0F1
                  - 8 * (term_E1E0_F0 + term_E1E0_F1 + term_E1_F0F1 + term_E0_F0F1)
                  + 4 * (term_E1_F0 + term_E1_F1 + term_E0_F0 + term_E0_F1)
                  + 4 * (term_E1E0 + term_F0F1)
                  - 2 * (term_E1 + term_E0 + term_F0 + term_F1)
                  + term_I)

    d_A1A0B1B0 = (16 * term_E1E0_F1F0
                  - 8 * (term_E1E0_F1 + term_E1E0_F0 + term_E1_F1F0 + term_E0_F1F0)
                  + 4 * (term_E1_F1 + term_E1_F0 + term_E0_F1 + term_E0_F0)
                  + 4 * (term_E1E0 + term_F1F0)
                  - 2 * (term_E1 + term_E0 + term_F1 + term_F0)
                  + term_I)

    d_A0A1A0B0B1B0 = (64 * term_E0E1E0_F0F1F0
                      - 32 * (term_E0E1E0_F0F1 + term_E0E1E0_F1F0 + term_E0E1_F0F1F0 + term_E1E0_F0F1F0)
                      + 16 * (
                              term_E0E1E0_F1 + term_E1_F0F1F0 + term_E0E1_F0F1 + term_E0E1_F1F0 + term_E1E0_F0F1 + term_E1E0_F1F0)
                      - 8 * (term_E0E1E0 + term_F0F1F0 + term_E0E1_F1 + term_E1E0_F1 + term_E1_F0F1 + term_E1_F1F0)
                      + 4 * (term_E0E1 + term_E1E0 + term_F0F1 + term_F1F0 + term_E1_F1)
                      - 2 * (term_E1 + term_F1)
                      + term_I)

    # --- 最终目标函数 ---
    part_1 = 0.5 + d_CHSH_std / (2 * np.sqrt(2))
    part_2 = - (1 / 8) * (d_A0A1B0B1 - d_A0A1B1B0 - d_A1A0B0B1 + d_A1A0B1B0)
    part_3 = (1 / (8 * np.sqrt(2))) * (
            3 * d_A1B1 - 2 * d_A0B1 - 2 * d_A1B0 + d_A0A1A0B1 - 2 * d_A0A1A0B0 + d_A1B0B1B0 - 2 * d_A0B0B1B0 - d_A0A1A0B0B1B0
    )

    return (part_1 + part_2 + part_3) / 2


# ==============================================================================
# 4. 核心计算逻辑
# ==============================================================================
def calculate_fidelity_at_point(target_val: float, coeff_matrix: np.ndarray, desc: np.ndarray, k: int = 3) -> float:
    m_vec = desc[2:4]

    # 1. 哑数据构建变量
    dummy_cg = np.zeros((3, 3))
    G, constraints, ind_catalog = npa_constraints(dummy_cg, desc, k=k, enforce_data=False)

    # 2. 施加约束: 输入的不等式(由coeff_matrix定义) == target_val
    bell_expr = build_general_bell_expression(G, ind_catalog, m_vec, coeff_matrix)
    constraints.append(bell_expr == target_val)

    # 3. 构造 Fidelity 目标
    func = build_fidelity_objective(G, ind_catalog, m_vec)

    # 4. 求解
    prob = cp.Problem(cp.Minimize(func), constraints)
    try:
        # 提高精度，增加迭代
        prob.solve(solver=cp.SCS, eps=1e-7, max_iters=20000)
    except cp.SolverError:
        return None

    if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        return prob.value
    else:
        return None


# ==============================================================================
# 5. 辅助绘图函数：计算并返回曲线数据
# ==============================================================================
def compute_curve(coeff_matrix, local_b, quant_b, num_points, desc, k):
    # [Task 2] 修改计算逻辑，基于 local_b 和 quant_b 生成扫描点
    values = np.linspace(local_b, quant_b, num_points)
    fidelities = []
    normalized_vals = []  # 用于存储归一化后的 V 值作为横坐标

    print(f"  Range [{local_b:.4f} -> {quant_b:.4f}]")

    for i, val in enumerate(values):
        # [Task 2] 计算归一化 V 值
        if quant_b - local_b != 0:
            V = (val - local_b) / (quant_b - local_b)
        else:
            V = 0.0

        # [Task 2] 打印 V 值
        print(f"    Point {i + 1}/{num_points}: Target={val:.4f} (V={V:.4f})...", end="")
        fid = calculate_fidelity_at_point(val, coeff_matrix, desc, k=k)

        if fid is not None:
            fid = max(0.0, min(1.0, fid))
            fidelities.append(fid)
            normalized_vals.append(V)  # [Task 2] 返回 V 作为横坐标
            print(f" Fid={fid:.4f}")
        else:
            print(" Infeasible")

    return normalized_vals, fidelities


# ==============================================================================
# 6. 主程序：绘图逻辑
# ==============================================================================
if __name__ == '__main__':
    # 配置
    desc = np.array([2, 2, 2, 2])
    k_level = 3
    num_points = 50  # 修改为 50 个点

    # --- 1. 计算 Standard CHSH 曲线 ---
    print("\n=== Computing Curve 1: Standard CHSH ===")
    coeff_chsh = np.array([
        [0, 0, 0],
        [0, 1, 1],
        [0, 1, -1]
    ])

    # [Task 1] 调用 compute_bell_limits 计算界限
    print("Computing bounds for CHSH...")
    local_bound1, quantum_bound1 = compute_bell_limits(coeff_chsh, desc, notation='fc', k=k_level)
    print(f"CHSH Bounds: Local={local_bound1:.4f}, Quantum={quantum_bound1:.4f}")

    # 调用 compute_curve，传入计算出的界限
    x1, y1 = compute_curve(coeff_chsh, local_bound1, quantum_bound1, num_points, desc, k_level)

    # --- 2. 计算 Inequality 2 曲线 ---
    print("\n=== Computing Curve 2: Inequality 2 ===")
    vec_ineq2 = [0., 0.565685, -0.4, 0.4, 1., 1., 1., -1.]
    coeff_ineq2 = convert_vector_to_matrix(vec_ineq2)

    # [Task 1] 调用 compute_bell_limits 计算界限
    print("Computing bounds for Inequality 2...")
    local_bound2, quantum_bound2 = compute_bell_limits(coeff_ineq2, desc, notation='fc', k=k_level)
    print(f"Ineq2 Bounds: Local={local_bound2:.4f}, Quantum={quantum_bound2:.4f}")

    # 调用 compute_curve，传入计算出的界限
    x2, y2 = compute_curve(coeff_ineq2, local_bound2, quantum_bound2, num_points, desc, k_level)

    # --- 3. 绘图 ---
    print("\nPlotting...")
    plt.figure(figsize=(10, 7))

    # [Task 2] 横轴现在是 V，范围大约在 0 到 1 之间
    plt.plot(x1, y1, marker='o', markersize=4, linestyle='-', color='blue', label='Standard CHSH')
    plt.plot(x2, y2, marker='s', markersize=4, linestyle='--', color='red', label='Inequality 2')

    # [Task 2] 标注重要界限 (归一化坐标系下，Local=0, Quantum=1)
    plt.axvline(x=0.0, color='gray', linestyle=':', alpha=0.5, label='Local Bound (V=0)')
    plt.axvline(x=1.0, color='green', linestyle=':', alpha=0.5, label='Quantum Bound (V=1)')

    plt.title(f'Robustness Comparison (NPA Level {k_level})')
    # [Task 2] 更新横轴标签
    plt.xlabel('Normalized Violation V = (Target - Local) / (Quantum - Local)')
    plt.ylabel('Minimum Fidelity F')
    plt.ylim(0, 1.05)
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.legend()

    # 保存图片
    figure_dir = os.path.join(parent_dir, 'figure')
    if not os.path.exists(figure_dir):
        os.makedirs(figure_dir)

    output_path = os.path.join(figure_dir, 'chsh_fidelity_figure3.png')

    plt.savefig(output_path, dpi=300)
    print(f"\nFigure saved to {output_path}")
    plt.show()