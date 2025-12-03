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
except ImportError as e:
    print(f"导入错误: {e}")
    print(f"请检查 NPAHierarchy_3.py 等文件是否存在于父目录: {parent_dir}")
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
def compute_curve(coeff_matrix, start_val, end_val, num_points, desc, k):
    values = np.linspace(start_val, end_val, num_points)
    fidelities = []
    valid_vals = []

    print(f"  Range [{start_val:.4f} -> {end_val:.4f}]")

    for i, val in enumerate(values):
        print(f"    Point {i + 1}/{num_points}: Target={val:.4f}...", end="")
        fid = calculate_fidelity_at_point(val, coeff_matrix, desc, k=k)

        if fid is not None:
            fid = max(0.0, min(1.0, fid))
            fidelities.append(fid)
            valid_vals.append(val)
            print(f" Fid={fid:.4f}")
        else:
            print(" Infeasible")

    return valid_vals, fidelities


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
    # CHSH 范围固定: 2.0 -> 2*sqrt(2)
    chsh_min = 2.0
    chsh_max = 2 * np.sqrt(2)

    x1, y1 = compute_curve(coeff_chsh, chsh_min, chsh_max, num_points, desc, k_level)

    # --- 2. 计算 Inequality 2 曲线 ---
    print("\n=== Computing Curve 2: Inequality 2 ===")
    vec_ineq2 = [0., 0.565685, -0.4, 0.4, 1., 1., 1., -1.]
    coeff_ineq2 = convert_vector_to_matrix(vec_ineq2)

    # 自动计算范围
    ineq2_min = calculate_classical_bound(coeff_ineq2)
    ineq2_max = 2 * np.sqrt(2)  # 固定为 2sqrt(2)

    x2, y2 = compute_curve(coeff_ineq2, ineq2_min, ineq2_max, num_points, desc, k_level)

    # --- 3. 绘图 ---
    print("\nPlotting...")
    plt.figure(figsize=(10, 7))


    # 为了在同一横坐标下对比，我们需要归一化 x 轴
    # 定义 Normalized Violation V = (Value - Local) / (Quantum - Local)
    # 这样 0 代表经典界，1 代表最大量子界

    def normalize(val_array, local_b, quant_b):
        return (np.array(val_array) - local_b) / (quant_b - local_b)


    # 绘制原始值 (Raw Values) 或者是 归一化值？
    # 你的要求是 "以val范围为横轴"，这意味着直接画原始值。
    # 但因为两个不等式的范围不同 (起点不同)，画在同一张图可能会错位。
    # 既然你要求画在同一图内，我就直接画原始值，各自有各自的曲线。

    plt.plot(x1, y1, marker='o', markersize=4, linestyle='-', color='blue', label='Standard CHSH')
    plt.plot(x2, y2, marker='s', markersize=4, linestyle='--', color='red', label='Inequality 2')

    # 标注重要界限
    plt.axvline(x=chsh_min, color='blue', linestyle=':', alpha=0.5, label='Local Bound (CHSH)')
    plt.axvline(x=ineq2_min, color='red', linestyle=':', alpha=0.5, label='Local Bound (Ineq 2)')
    plt.axvline(x=2 * np.sqrt(2), color='green', linestyle='-', alpha=0.5, label='Quantum Bound')

    plt.title(f'Robustness Comparison (NPA Level {k_level})')
    plt.xlabel('Inequality Violation Value')
    plt.ylabel('Minimum Fidelity F')
    plt.ylim(0, 1.05)
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.legend()

    # 保存图片
    figure_dir = os.path.join(parent_dir, 'figure')
    if not os.path.exists(figure_dir):
        os.makedirs(figure_dir)

    output_path = os.path.join(figure_dir, 'chsh_fidelity_figure2.png')

    plt.savefig(output_path, dpi=300)
    print(f"\nFigure saved to {output_path}")
    plt.show()