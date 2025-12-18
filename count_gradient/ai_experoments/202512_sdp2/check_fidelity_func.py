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
    返回对应的态矢量
    """
    # 基矢 |00> = [1, 0, 0, 0]^T
    ket_00 = np.array([1, 0, 0, 0], dtype=complex)
    # 基矢 |11> = [0, 0, 0, 1]^T
    ket_11 = np.array([0, 0, 0, 1], dtype=complex)

    psi = np.cos(theta) * ket_00 + np.sin(theta) * ket_11
    return psi


def get_base_observables(mu):
    """
    根据物理定义构建基础可观测量矩阵 (4x4)
    Alice: A0=Z, A1=X
    Bob:   B0=cos(mu)Z + sin(mu)X, B1=cos(mu)Z - sin(mu)X
    """
    I2, X, Z = get_pauli_matrices()

    # Alice Operators (acting on qubit 1)
    # A0 = Z (x) I
    A0_mat = np.kron(Z, I2)
    # A1 = X (x) I
    A1_mat = np.kron(X, I2)

    # Bob Operators (acting on qubit 2)
    # B0_local = cos(mu)Z + sin(mu)X
    B0_local = np.cos(mu) * Z + np.sin(mu) * X
    # B0 = I (x) B0_local
    B0_mat = np.kron(I2, B0_local)

    # B1_local = cos(mu)Z - sin(mu)X
    B1_local = np.cos(mu) * Z - np.sin(mu) * X
    # B1 = I (x) B1_local
    B1_mat = np.kron(I2, B1_local)

    # Identity (4x4)
    Identity_mat = np.eye(4, dtype=complex)

    # 存入字典方便调用
    obs_dict = {
        "I": Identity_mat,
        "A0": A0_mat,
        "A1": A1_mat,
        "B0": B0_mat,
        "B1": B1_mat
    }

    return obs_dict


def compute_operator_product(op_string, obs_dict):
    """
    解析算符字符串 (如 "A0 B1 A0") 并计算对应的矩阵乘积
    """
    if not op_string or op_string.strip() == "I":
        return obs_dict["I"]

    # 分割字符串，例如 "A0 B1" -> ["A0", "B1"]
    ops_keys = op_string.split()

    # 初始矩阵为单位阵
    result_mat = np.eye(4, dtype=complex)

    # 依次左乘 (注意矩阵乘法顺序: A B C -> A @ B @ C)
    for key in ops_keys:
        if key in obs_dict:
            result_mat = result_mat @ obs_dict[key]
        else:
            raise ValueError(f"未知的算符键: {key}")

    return result_mat


def calculate_expectation(psi, operator_matrix):
    """
    计算期望值 <psi|O|psi>
    """
    # psi.conj().T 是 bra <psi|
    # dot product: <psi| * (O * |psi>)
    return np.vdot(psi, operator_matrix @ psi).real


# ==========================================
# 2. 核心计算逻辑
# ==========================================

def verify_fidelity_from_json(json_path, theta_val, mu_val):
    """
    读取JSON，计算各项期望值与系数的乘积之和
    """
    # 1. 检查文件是否存在
    if not os.path.exists(json_path):
        print(f"错误: 找不到文件 {json_path}")
        return None

    # 2. 读取 JSON
    with open(json_path, 'r', encoding='utf-8') as f:
        try:
            coeffs_data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"JSON 解析错误: {e}")
            return None

    print(f"成功读取 JSON 文件: {json_path}")
    print(f"包含 {len(coeffs_data)} 项。")

    # 3. 准备量子态和基础算符
    psi = get_quantum_state(theta_val)
    obs_dict = get_base_observables(mu_val)

    total_sum = 0.0

    # 4. 遍历 JSON 计算
    # 准备 eval 的上下文环境，包含 numpy 函数和变量
    eval_context = {
        "np": np,
        "cos": np.cos,
        "sin": np.sin,
        "theta": theta_val,
        "mu": mu_val
    }

    print("-" * 70)
    print(f"{'Observable Combination':<30} | {'Coeff Value':<15} | {'Exp Value':<15}")
    print("-" * 70)

    for key, coeff_str in coeffs_data.items():
        # A. 解析算符组合并计算矩阵
        try:
            op_matrix = compute_operator_product(key, obs_dict)
        except Exception as e:
            print(f"算符构建失败 [{key}]: {e}")
            continue

        # B. 计算量子期望值 <psi| Op |psi>
        exp_val = calculate_expectation(psi, op_matrix)

        # C. 计算系数数值 (解析字符串表达式)
        try:
            # 清洗字符串，处理可能的换行或非标准字符
            clean_coeff_str = coeff_str.replace('\n', '').strip()
            coeff_val = eval(clean_coeff_str, eval_context)
        except Exception as e:
            print(f"系数计算失败 [{key}]: {e} (Expr: {coeff_str})")
            continue

        # D. 累加
        term_contribution = coeff_val * exp_val
        total_sum += term_contribution

        # 打印非零项用于调试
        if abs(term_contribution) > 1e-6:
            print(f"{key:<30} | {coeff_val: .6f}        | {exp_val: .6f}")

    print("-" * 70)
    return total_sum


# ==========================================
# 3. 主程序入口 (变量设置)
# ==========================================

if __name__ == "__main__":
    # --- 变量设置 ---
    # 设定 theta
    theta_val = np.pi / 6  # 30 degrees

    # 根据关系式 tan(mu) = sin(2*theta) 计算 mu
    # sin(2*pi/6) = sin(pi/3) = sqrt(3)/2
    # mu = arctan(sqrt(3)/2)
    sin_2theta = np.sin(2 * theta_val)
    mu_val = np.arctan(sin_2theta)

    print(f"参数设置:")
    print(f"theta = {theta_val:.6f} (rad) = {np.degrees(theta_val):.2f} (deg)")
    print(f"mu    = {mu_val:.6f} (rad) = {np.degrees(mu_val):.2f} (deg)")
    print(f"验证关系 tan(mu) = {np.tan(mu_val):.4f}, sin(2theta) = {sin_2theta:.4f}")

    # --- 路径设置 (修改部分) ---
    json_filename = "fidelity_coeffs_obs.json"

    # 获取当前脚本所在的绝对目录
    try:
        current_script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        # 如果在某些交互式环境中 __file__ 未定义，则使用当前工作目录
        current_script_dir = os.getcwd()

    # 构建目标路径：当前脚本目录/func/fidelity_coeffs_obs.json
    target_path = os.path.join(current_script_dir, "func", json_filename)

    # 执行计算
    if os.path.exists(target_path):
        result = verify_fidelity_from_json(target_path, theta_val, mu_val)

        if result is not None:
            print(f"\n========================================")
            print(f"最终计算结果 (Fidelity): {result:.8f}")
            print(f"========================================")
    else:
        print(f"\n错误：找不到文件 {target_path}")
        print(f"请确保该文件位于当前脚本目录下的 'func' 子文件夹中。")