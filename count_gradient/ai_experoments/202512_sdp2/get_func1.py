import sympy
from sympy import symbols, expand, Mul, Add, simplify
import json
import os
import numpy as np


# ==========================================
# 1. 核心化简逻辑 (Observables)
# ==========================================

def get_adjoint(expr):
    """
    计算非对易多项式的厄米共轭 (Hermitian Conjugate)。
    假设系数为实数。规则: (A B C)^dag = C^dag B^dag A^dag
    注意: 这里的 A, B, C 指的是 A0, B0 等厄米可观测量。
    """
    expr = expand(expr)
    terms = Add.make_args(expr)
    new_terms = []

    for term in terms:
        c_factors, nc_factors = term.args_cnc()
        # 系数取共轭 (假设实数则不变)
        coeff = Mul(*c_factors)

        # 反转非对易算符的顺序
        rev_nc = list(reversed(nc_factors))

        new_terms.append(coeff * Mul(*rev_nc))

    return Add(*new_terms)


def reduce_observable_indices(indices):
    """
    化简可观测量序列。
    规则: A^2 = I, 且不同方的算符对易。
    """

    def get_party(idx):
        return 0 if idx < 2 else 1

    while True:
        made_simplification = False
        length = len(indices)
        if length == 0: break

        i = 0
        while i < len(indices) - 1:
            j = i + 1
            idx_i = indices[i]
            idx_j = indices[j]
            party_i = get_party(idx_i)
            party_j = get_party(idx_j)

            # Rule 1: A^2 = I (抵消)
            if idx_i == idx_j:
                indices.pop(j)
                indices.pop(i)
                made_simplification = True
                break

            # Rule 2: 同方阻隔 (不能跨越)
            if party_i == party_j:
                i += 1
                continue

            # Rule 3: 对易性 (不同方可交换)
            if idx_i > idx_j:
                indices[i], indices[j] = indices[j], indices[i]
                made_simplification = True
                break

            i += 1
        if not made_simplification: break

    return indices


def simplify_observables_advanced(expr, ops_dict):
    A0, A1, B0, B1 = ops_dict['A0'], ops_dict['A1'], ops_dict['B0'], ops_dict['B1']
    sym_to_int = {A0: 0, A1: 1, B0: 2, B1: 3}
    int_to_sym = {0: A0, 1: A1, 2: B0, 3: B1}

    expanded_expr = expand(expr)
    terms = Add.make_args(expanded_expr)
    final_terms = []

    for term in terms:
        c_factors, nc_factors = term.args_cnc()
        coeff = Mul(*c_factors)

        if not nc_factors:
            final_terms.append(coeff)
            continue

        indices = []
        for factor in nc_factors:
            # 处理幂次 A^n
            if factor.is_Pow and factor.base in sym_to_int:
                if factor.exp % 2 == 1:
                    indices.append(sym_to_int[factor.base])
            elif factor in sym_to_int:
                indices.append(sym_to_int[factor])

        simplified_indices = reduce_observable_indices(indices)

        if not simplified_indices:
            final_terms.append(coeff)
        else:
            ops = [int_to_sym[idx] for idx in simplified_indices]
            final_terms.append(coeff * Mul(*ops))

    return Add(*final_terms)


# ==========================================
# 2. 核心化简逻辑 (Projectors) - [从原错误代码移植]
# ==========================================

def reduce_projector_indices(indices):
    """
    化简投影算子序列。
    规则: E^2 = E, 且不同方的算符对易。
    """

    def get_party(idx):
        return 0 if idx < 2 else 1

    while True:
        made_simplification = False
        length = len(indices)

        if length == 0:
            break

        i = 0
        while i < len(indices) - 1:
            j = i + 1
            idx_i = indices[i]
            idx_j = indices[j]
            party_i = get_party(idx_i)
            party_j = get_party(idx_j)

            # 规则: E^2 = E (合并，注意这里只 pop 一个)
            if idx_i == idx_j:
                indices.pop(j)
                made_simplification = True
                break

            # 规则: 同方阻隔
            if party_i == party_j:
                i += 1
                continue

            # 规则: 交换顺序
            if idx_i > idx_j:
                indices[i], indices[j] = indices[j], indices[i]
                made_simplification = True
                break

            i += 1

        if not made_simplification:
            break

    return indices


def simplify_projectors_advanced(expr, proj_dict):
    E0, E1, F0, F1 = proj_dict['E0'], proj_dict['E1'], proj_dict['F0'], proj_dict['F1']
    sym_to_int = {E0: 0, E1: 1, F0: 2, F1: 3}
    int_to_sym = {0: E0, 1: E1, 2: F0, 3: F1}

    expanded_expr = expand(expr)
    terms = Add.make_args(expanded_expr)
    final_terms = []

    for term in terms:
        c_factors, nc_factors = term.args_cnc()
        coeff = Mul(*c_factors)

        if not nc_factors:
            final_terms.append(coeff)
            continue

        indices = []
        for factor in nc_factors:
            if factor.is_Pow:
                base = factor.base
                exp = factor.exp
                if base in sym_to_int and exp.is_Integer and exp > 0:
                    idx = sym_to_int[base]
                    # E^n = E, 所以只需要添加一个
                    indices.append(idx)
            else:
                if factor in sym_to_int:
                    indices.append(sym_to_int[factor])

        simplified_indices = reduce_projector_indices(indices)

        if not simplified_indices:
            final_terms.append(coeff)
        else:
            ops = [int_to_sym[idx] for idx in simplified_indices]
            final_terms.append(coeff * Mul(*ops))

    return Add(*final_terms)


# ==========================================
# 3. 生成逻辑 (Generation)
# ==========================================
def generate_and_verify():
    # 定义符号
    A0, A1 = symbols('A0 A1', commutative=False)
    B0, B1 = symbols('B0 B1', commutative=False)
    ops_dict = {'A0': A0, 'A1': A1, 'B0': B0, 'B1': B1}

    # 投影算子符号
    E0, E1 = symbols('E0 E1', commutative=False)
    F0, F1 = symbols('F0 F1', commutative=False)
    proj_dict = {'E0': E0, 'E1': E1, 'F0': F0, 'F1': F1}

    theta, mu = symbols('theta mu', real=True)
    Identity = 1

    print("=== 1. 构建 SWAP 算符 ===")
    # Alice 算符 (作用在物理空间)
    vec_A = [(Identity + A0) / 2, A1 * (Identity - A0) / 2]

    # Bob 算符
    Z_B = (B0 + B1) / (2 * sympy.cos(mu))
    X_B = (B0 - B1) / (2 * sympy.sin(mu))
    vec_B = [(Identity + Z_B) / 2, X_B * (Identity - Z_B) / 2]

    # Kraus 算符 M_k (Indices: 0->00, 3->11)
    # Omega[0] = M_00, Omega[3] = M_11
    Omega = []
    for op_a in vec_A:
        for op_b in vec_B:
            Omega.append(op_a * op_b)

    # 目标态系数
    c = sympy.cos(theta)
    s = sympy.sin(theta)

    # === 关键修正点 ===
    # 构建有效算符 M_eff = <bar_psi | Psi_total>
    # F = <psi | M_eff^dag M_eff | psi>
    print("=== 2. 计算保真度算符 (使用 Correct M^dag M) ===")

    M_eff = c * Omega[0] + s * Omega[3]

    # 计算共轭转置
    M_eff_dag = get_adjoint(M_eff)

    # 得到保真度算符 P (Observables 形式)
    F_op_raw = M_eff_dag * M_eff

    # 化简 (Observables)
    F_op = simplify_observables_advanced(F_op_raw, ops_dict)

    # 提取系数 (Observables)
    terms_obs = Add.make_args(F_op)
    coeff_dict_obs = {}
    for term in terms_obs:
        c_factors, nc_factors = term.args_cnc()
        coeff = Mul(*c_factors)
        key = "I" if not nc_factors else " ".join([str(f) for f in nc_factors])

        if key in coeff_dict_obs:
            coeff_dict_obs[key] += coeff
        else:
            coeff_dict_obs[key] = coeff

    # 清洗并输出 Observables JSON
    final_json = {}
    for k, v in coeff_dict_obs.items():
        val_simp = simplify(v)
        if val_simp != 0:
            # 强制去除换行符，并替换 numpy 函数名
            s_val = str(val_simp).replace('\n', '').replace(' ', '')
            s_val = s_val.replace('sin', 'np.sin').replace('cos', 'np.cos')
            final_json[k] = s_val

    # 路径处理
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        current_dir = os.getcwd()
    target_dir = os.path.join(current_dir, 'func')
    if not os.path.exists(target_dir): os.makedirs(target_dir)

    file_path_obs = os.path.join(target_dir, 'fidelity_coeffs_obs.json')

    with open(file_path_obs, 'w') as f:
        json.dump(final_json, f, indent=2)
    print(f"Observables JSON 文件已生成至: {file_path_obs}")

    # ==========================================
    # 3. 验证逻辑 (Verification) - 对 Observables 形式验证
    # ==========================================
    print("\n=== 3. 运行数值验证 (Observables, theta=pi/6) ===")

    theta_val = np.pi / 6
    mu_val = np.arctan(np.sin(2 * theta_val))

    # 定义矩阵
    I_mat = np.eye(2)
    X_mat = np.array([[0, 1], [1, 0]])
    Z_mat = np.array([[1, 0], [0, -1]])

    A0_mat = np.kron(Z_mat, I_mat)
    A1_mat = np.kron(X_mat, I_mat)

    B0_l = np.cos(mu_val) * Z_mat + np.sin(mu_val) * X_mat
    B1_l = np.cos(mu_val) * Z_mat - np.sin(mu_val) * X_mat
    B0_mat = np.kron(I_mat, B0_l)
    B1_mat = np.kron(I_mat, B1_l)

    obs_map = {"I": np.eye(4), "A0": A0_mat, "A1": A1_mat, "B0": B0_mat, "B1": B1_mat}

    psi = np.cos(theta_val) * np.array([1, 0, 0, 0]) + np.sin(theta_val) * np.array([0, 0, 0, 1])

    total = 0
    print("-" * 50)
    print(f"{'Key':<25} | {'Val':<10}")
    print("-" * 50)

    for k, v_str in final_json.items():
        # Eval coeff
        ctx = {"np": np, "theta": theta_val, "mu": mu_val}
        coeff = eval(v_str, ctx)

        # Matrix Prod
        keys = k.split()
        mat = np.eye(4)
        if k != "I":
            for op in keys:
                mat = mat @ obs_map[op]

        # Expectation
        exp = np.vdot(psi, mat @ psi).real
        term_val = coeff * exp
        total += term_val

        # 输出所有项（去掉了阈值判断）
        print(f"{k:<25} | {term_val:.8f}")

    print("-" * 50)
    print(f"Observables 验证结果 (Fidelity): {total:.8f}")

    # ==========================================
    # 4. 转换为投影算子形式并生成 JSON
    # ==========================================
    print("\n=== 4. 正在转换为投影算子形式并展开 ===")

    # 替换规则: A = 2E - I
    subs_dict = {
        A0: 2 * E0 - 1,
        A1: 2 * E1 - 1,
        B0: 2 * F0 - 1,
        B1: 2 * F1 - 1
    }

    # 注意：这里使用的是经过化简的 Correct Observables Form (F_op)
    F_proj_raw = F_op.subs(subs_dict)

    print("=== 5. 调用投影算子化简逻辑 (Idempotency & Sorting) ===")
    F_proj = simplify_projectors_advanced(F_proj_raw, proj_dict)

    print("Projectors 化简结果预览:")
    print(str(F_proj)[:200] + "...")

    print("=== 6. 提取 Projectors 系数并生成 JSON ===")
    terms_proj = Add.make_args(F_proj)
    coeff_dict_proj = {}

    for term in terms_proj:
        c_factors, nc_factors = term.args_cnc()
        coeff = Mul(*c_factors)
        key = "I" if not nc_factors else " ".join([str(f) for f in nc_factors])

        if key in coeff_dict_proj:
            coeff_dict_proj[key] += coeff
        else:
            coeff_dict_proj[key] = coeff

    # 清洗并输出 Projectors JSON
    final_json_proj = {}
    for k, v in coeff_dict_proj.items():
        val_simp = simplify(v)
        if val_simp != 0:
            s_val = str(val_simp).replace('\n', '').replace(' ', '')
            s_val = s_val.replace('sin', 'np.sin').replace('cos', 'np.cos')
            final_json_proj[k] = s_val

    file_path_proj = os.path.join(target_dir, 'fidelity_coeffs.json')

    with open(file_path_proj, 'w') as f:
        json.dump(final_json_proj, f, indent=2)

    print(f"Projectors JSON 文件已生成至: {file_path_proj}")

    # 简单预览
    print("Projectors JSON 内容预览 (前 5 项):")
    count = 0
    for k, v in final_json_proj.items():
        print(f'  "{k}": "{v}"')
        count += 1
        if count >= 5: break
    print("  ...")


if __name__ == "__main__":
    generate_and_verify()