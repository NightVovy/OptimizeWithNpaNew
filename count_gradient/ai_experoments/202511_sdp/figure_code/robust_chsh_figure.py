import numpy as np
import cvxpy as cp
import sys
import os
import matplotlib.pyplot as plt

# ==============================================================================
# [修改点 1] 设置路径：将父目录加入系统路径，以便导入模块
# ==============================================================================
# 获取当前脚本所在的绝对路径 (.../figure_code)
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取父目录路径 (.../)
parent_dir = os.path.dirname(current_dir)

# 将父目录加入 Python 搜索路径
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# ==============================================================================
# [修改点 2] 导入依赖 (现在可以正常从父目录导入了)
# ==============================================================================
try:
    from NPAHierarchy_3 import npa_constraints
    from locate_gamma_2 import get_gamma_element
except ImportError as e:
    print(f"导入错误: {e}")
    print(f"请检查 NPAHierarchy_3.py 等文件是否存在于父目录: {parent_dir}")
    sys.exit(1)


# ==============================================================================
# 1. 辅助函数：构建 Bell 表达式约束
# ==============================================================================
def build_general_bell_expression(G, ind_catalog, m_vec, coeff_matrix):
    expr = 0
    # 1. 常数项
    if coeff_matrix[0, 0] != 0:
        expr += coeff_matrix[0, 0]

    def get_term_E(alice_list, bob_list):
        t = get_gamma_element(G, ind_catalog, m_vec, alice_list, bob_list)
        if t is None:
            raise ValueError(f"无法在 Gamma 矩阵中找到算符 A{alice_list}B{bob_list}。")
        return t

    # 2. Bob 单体项
    for y_idx in range(2):
        c = coeff_matrix[0, y_idx + 1]
        if c != 0:
            bob_setting = [2 + y_idx]
            term_Fy = get_term_E([], bob_setting)
            expr += c * (2 * term_Fy - 1)

    # 3. Alice 单体项
    for x_idx in range(2):
        c = coeff_matrix[x_idx + 1, 0]
        if c != 0:
            term_Ex = get_term_E([x_idx], [])
            expr += c * (2 * term_Ex - 1)

    # 4. 关联项
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
# 2. 辅助函数：构建 Fidelity 目标函数
# ==============================================================================
def build_fidelity_objective(G, ind_catalog, m_vec, chsh_val_for_func):
    def get_term(alice_seq, bob_seq):
        val = get_gamma_element(G, ind_catalog, m_vec, alice_seq, bob_seq)
        if val is None:
            raise ValueError(f"Term <A{alice_seq} B{bob_seq}> not found.")
        return val

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
    d_A1B1 = 4 * term_E1_F1 - 2 * term_E1 - 2 * term_F1 + term_I
    d_A0B1 = 4 * term_E0_F1 - 2 * term_E0 - 2 * term_F1 + term_I
    d_A1B0 = 4 * term_E1_F0 - 2 * term_E1 - 2 * term_F0 + term_I

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

    # --- 6-body term ---
    d_A0A1A0B0B1B0 = (64 * term_E0E1E0_F0F1F0
                      - 32 * (term_E0E1E0_F0F1 + term_E0E1E0_F1F0 + term_E0E1_F0F1F0 + term_E1E0_F0F1F0)
                      + 16 * (
                              term_E0E1E0_F1 + term_E1_F0F1F0 + term_E0E1_F0F1 + term_E0E1_F1F0 + term_E1E0_F0F1 + term_E1E0_F1F0)
                      - 8 * (term_E0E1E0 + term_F0F1F0 + term_E0E1_F1 + term_E1E0_F1 + term_E1_F0F1 + term_E1_F1F0)
                      + 4 * (term_E0E1 + term_E1E0 + term_F0F1 + term_F1F0 + term_E1_F1)
                      - 2 * (term_E1 + term_F1)
                      + term_I)

    part_1 = 0.5 + chsh_val_for_func / (2 * np.sqrt(2))
    part_2 = - (1 / 8) * (d_A0A1B0B1 - d_A0A1B1B0 - d_A1A0B0B1 + d_A1A0B1B0)
    part_3 = (1 / (8 * np.sqrt(2))) * (
            3 * d_A1B1 - 2 * d_A0B1 - 2 * d_A1B0 + d_A0A1A0B1 - 2 * d_A0A1A0B0 + d_A1B0B1B0 - 2 * d_A0B0B1B0 - d_A0A1A0B0B1B0
    )

    return (part_1 + part_2 + part_3) / 2


# ==============================================================================
# 3. 核心计算逻辑
# ==============================================================================
def calculate_fidelity_at_point(target_val: float, coeff_matrix: np.ndarray, desc: np.ndarray, k: int = 3) -> float:
    m_vec = desc[2:4]

    dummy_cg = np.zeros((3, 3))
    G, constraints, ind_catalog = npa_constraints(dummy_cg, desc, k=k, enforce_data=False)

    bell_expr = build_general_bell_expression(G, ind_catalog, m_vec, coeff_matrix)
    constraints.append(bell_expr == target_val)

    func = build_fidelity_objective(G, ind_catalog, m_vec, target_val)

    prob = cp.Problem(cp.Minimize(func), constraints)
    try:
        prob.solve(solver=cp.SCS, eps=1e-6)
    except cp.SolverError:
        return None

    if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        return prob.value
    else:
        return None


# ==============================================================================
# 4. 主程序：绘图逻辑
# ==============================================================================
if __name__ == '__main__':
    # 配置
    desc = np.array([2, 2, 2, 2])
    k_level = 3
    num_points = 20

    # 定义范围: 2.0 到 2*sqrt(2)
    min_val = 2.0
    max_val = 2 * np.sqrt(2)
    chsh_values = np.linspace(min_val, max_val, num_points)

    # 定义 CHSH 系数
    coeff_chsh = np.array([
        [0, 0, 0],
        [0, 1, 1],
        [0, 1, -1]
    ])

    print(f"Calculating Fidelity Curve ({num_points} points)...")
    print(f"Range: [{min_val:.4f}, {max_val:.4f}]")

    fidelities = []
    valid_chsh = []

    for i, val in enumerate(chsh_values):
        print(f"Processing point {i + 1}/{num_points}: CHSH={val:.4f}...", end="")
        fid = calculate_fidelity_at_point(val, coeff_chsh, desc, k=k_level)

        if fid is not None:
            fid = max(0.0, min(1.0, fid))
            fidelities.append(fid)
            valid_chsh.append(val)
            print(f" Done. Fidelity={fid:.4f}")
        else:
            print(" Failed (Infeasible).")

    # 绘图
    plt.figure(figsize=(8, 6))
    plt.plot(valid_chsh, fidelities, marker='o', linestyle='-', color='b', label='CHSH Self-Testing')
    plt.plot([max_val], [1.0], marker='*', color='r', markersize=10, label='Tsirelson Bound (F=1)')

    plt.title(f'Robustness of CHSH Self-Testing (NPA Level {k_level})')
    plt.xlabel('CHSH Violation Value')
    plt.ylabel('Minimum Fidelity F')
    plt.xlim(1.9, 2.9)
    plt.ylim(0, 1.05)
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.legend()

    # ==============================================================================
    # [修改点 3] 设置图片保存路径
    # ==============================================================================
    # 构造 ../figure/ 目录的绝对路径
    figure_dir = os.path.join(parent_dir, 'figure')

    # 如果目录不存在，创建它
    if not os.path.exists(figure_dir):
        os.makedirs(figure_dir)

    output_path = os.path.join(figure_dir, 'chsh_fidelity_figure1.png')

    plt.savefig(output_path, dpi=300)
    print(f"\nFigure saved to {output_path}")
    plt.show()