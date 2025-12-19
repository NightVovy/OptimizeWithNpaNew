import json
import numpy as np
import os


# ==========================================
# 1. 基础量子力学函数定义
# ==========================================

def get_pauli_matrices():
    """定义泡利矩阵"""
    I = np.eye(2)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    return I, X, Z


def get_quantum_state(theta):
    """
    生成量子态 psi = cos(theta)|00> + sin(theta)|11>
    """
    ket_00 = np.array([1, 0, 0, 0], dtype=complex)
    ket_11 = np.array([0, 0, 0, 1], dtype=complex)
    psi = np.cos(theta) * ket_00 + np.sin(theta) * ket_11
    return psi


def get_projectors(mu):
    """
    构建投影算子矩阵 (4x4 扩展形式)
    Alice: E0, E1
    Bob:   F0, F1
    """
    I2, X, Z = get_pauli_matrices()

    # --- Alice 的投影算子 (作用在 Qubit 1) ---
    # A0 = Z => E0 = (I+Z)/2 = |0><0|
    E0_local = (I2 + Z) / 2
    # A1 = X => E1 = (I+X)/2 = |+><+|
    E1_local = (I2 + X) / 2

    # 扩展到 4x4 (E x I)
    E0 = np.kron(E0_local, I2)
    E1 = np.kron(E1_local, I2)

    # --- Bob 的投影算子 (作用在 Qubit 2) ---
    # B0 = cos(mu)Z + sin(mu)X => F0 = (I+B0)/2
    B0_local = np.cos(mu) * Z + np.sin(mu) * X
    F0_local = (I2 + B0_local) / 2

    # B1 = cos(mu)Z - sin(mu)X => F1 = (I+B1)/2
    B1_local = np.cos(mu) * Z - np.sin(mu) * X
    F1_local = (I2 + B1_local) / 2

    # 扩展到 4x4 (I x F)
    F0 = np.kron(I2, F0_local)
    F1 = np.kron(I2, F1_local)

    # 4x4 单位阵
    Identity = np.eye(4, dtype=complex)

    # 存入字典
    proj_map = {
        "I": Identity,
        "E0": E0,
        "E1": E1,
        "F0": F0,
        "F1": F1
    }

    return proj_map


def compute_term_matrix(key_string, proj_map):
    """
    解析投影算子组合字符串 (如 "E0 F0 E1") 并计算矩阵乘积
    注意：这里的乘法是矩阵乘法 (dot product)，顺序很重要
    """
    if not key_string or key_string.strip() == "I":
        return proj_map["I"]

    keys = key_string.split()

    # 初始为单位阵
    res_mat = np.eye(4, dtype=complex)

    for k in keys:
        if k in proj_map:
            # 左乘：新的算符在右边累乘 (Op1 @ Op2 @ Op3)
            # 因为 JSON 中的顺序即为算符作用顺序
            res_mat = res_mat @ proj_map[k]
        else:
            raise ValueError(f"未知的投影算子键: {k}")

    return res_mat


# ==========================================
# 2. 验证主逻辑
# ==========================================

def verify_fidelity_projectors():
    # --- A. 参数设置 ---
    theta_val = np.pi / 6
    # 根据关系 tan(mu) = sin(2*theta)
    mu_val = np.arctan(np.sin(2 * theta_val))

    print(f"=== 参数设置 ===")
    print(f"theta = {theta_val:.6f} rad")
    print(f"mu    = {mu_val:.6f} rad")

    # --- B. 获取物理对象 ---
    psi = get_quantum_state(theta_val)
    proj_map = get_projectors(mu_val)

    # --- C. 读取 JSON ---
    json_filename = "fidelity_coeffs.json"
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        current_dir = os.getcwd()

    target_path = os.path.join(current_dir, "func", json_filename)

    if not os.path.exists(target_path):
        print(f"错误: 找不到文件 {target_path}")
        return

    with open(target_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"\n成功读取 JSON 文件: {target_path}")
    print(f"包含 {len(data)} 项。")

    # --- D. 计算求和 ---
    total_fidelity = 0.0

    # 准备 eval 上下文
    ctx = {
        "np": np,
        "sin": np.sin,
        "cos": np.cos,
        "theta": theta_val,
        "mu": mu_val
    }

    print(f"\n{'Projector Key':<30} | {'Coeff':<12} | {'Exp Value':<12} | {'Term Contribution'}")
    print("-" * 80)

    for key, coeff_str in data.items():
        # 1. 计算系数数值
        try:
            # 移除可能的换行符
            clean_str = coeff_str.replace('\n', '').strip()
            coeff_val = eval(clean_str, ctx)
        except Exception as e:
            print(f"系数解析错误 [{key}]: {e}")
            continue

        # 2. 计算算符矩阵
        mat = compute_term_matrix(key, proj_map)

        # 3. 计算期望值 <psi| Op |psi>
        # 注意: 投影算子的乘积不一定是厄米的，但保真度分解通常保证总和是实数
        # 这里我们取实部
        exp_val = np.vdot(psi, mat @ psi).real

        # 4. 累加
        term_val = coeff_val * exp_val
        total_fidelity += term_val

        # 只打印贡献较大的项以便检查
        if abs(term_val) > 1e-4:
            print(f"{key:<30} | {coeff_val: .4f}     | {exp_val: .4f}     | {term_val: .6f}")

    print("-" * 80)
    print(f"\n>>> 最终验证结果 (Fidelity): {total_fidelity:.10f}")

    # 理论值检查
    if abs(total_fidelity - 1.0) < 1e-6:
        print(">>> 验证成功！结果等于 1.0")
    else:
        print(f">>> 验证偏差: {abs(total_fidelity - 1.0)}")


if __name__ == "__main__":
    verify_fidelity_projectors()