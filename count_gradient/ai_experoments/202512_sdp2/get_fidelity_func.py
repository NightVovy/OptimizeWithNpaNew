import sympy
from sympy import symbols, I, expand, Mul, Add, simplify
import json
import os
import numpy as np


# ==========================================
# 新增：可观测量化简核心算法 (A^2 = I)
# ==========================================
def reduce_observable_indices(indices):
    """
    针对可观测量的化简逻辑。
    输入: 整数列表 (例如 [0, 0, 2, 3])
    输出: 化简后的整数列表 (例如 [2, 3]) -> 因为 0和0 抵消为 I
    映射: A0->0, A1->1, B0->2, B1->3
    """

    def get_party(idx):
        # 0, 1 是 Alice (0); 2, 3 是 Bob (1)
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

            # --- 规则 1: 幂等性/对合性 (Involution) A^2 = I ---
            # 这里的区别是: 投影算子合并(pop一个)，可观测量抵消(pop两个)
            if idx_i == idx_j:
                # 删除 j 和 i (先删后面的 j 以免影响 i 的索引)
                indices.pop(j)
                indices.pop(i)
                made_simplification = True
                break  # 结构剧烈变化，跳出重来

            # --- 规则 2: 同方阻隔 (Same party barrier) ---
            # 如果属于同一方，但不是同一个算符 (例如 A0 和 A1)，无法交换
            # 必须停止对当前 i 的后续 j 搜索。
            # 直接跳过当前 i，处理下一个 i
            if party_i == party_j:
                i += 1
                continue

            # --- 规则 3: 交换顺序 (Commutativity) ---
            # 如果属于不同方，且 Bob (大索引) 在 Alice (小索引) 前面 -> 交换
            if idx_i > idx_j:
                indices[i], indices[j] = indices[j], indices[i]
                made_simplification = True
                break

                # 如果无事发生，继续检查下一个 j (但在本逻辑结构中，
            # j 是通过 i 的移动来隐式重置的，这里 i 不变继续循环是不对的，
            # 应该通过 i+=1 来控制。但上面的 barrier 逻辑已经处理了同方。
            # 对于不同方且顺序正确的情况 (A B)，我们需要继续检查 A 是否能和 B 后面的 A 作用吗?
            # 比如 A0 B0 A0。
            # i=0(A0), j=1(B0). party不同, 顺序正确.
            # 我们需要继续增加 j 吗?
            # 当前 while 结构是 j = i+1 固定。
            # 参考逻辑通常是双重循环。但在 Python list 变长操作中，
            # 我们使用了 restart strategy (break outer while)。
            # 所以这里如果没有 break，应该让 i 增加。
            i += 1

        if not made_simplification:
            break

    return indices


def simplify_observables_advanced(expr, ops_dict):
    """
    新的可观测量化简入口函数。
    """
    A0, A1, B0, B1 = ops_dict['A0'], ops_dict['A1'], ops_dict['B0'], ops_dict['B1']

    # 建立映射表
    sym_to_int = {A0: 0, A1: 1, B0: 2, B1: 3}
    int_to_sym = {0: A0, 1: A1, 2: B0, 3: B1}

    # 展开表达式
    expanded_expr = expand(expr)
    terms = Add.make_args(expanded_expr)
    final_terms = []

    for term in terms:
        c_factors, nc_factors = term.args_cnc()
        coeff = Mul(*c_factors)

        if not nc_factors:
            final_terms.append(coeff)
            continue

        # 步骤 A: 转换为整数列表
        indices = []
        for factor in nc_factors:
            if factor.is_Pow:
                base = factor.base
                exp = factor.exp
                # 处理 A^n，如果 n 是偶数则抵消为 I，奇数则保留一个 A
                if base in sym_to_int and exp.is_Integer:
                    if exp % 2 == 1:
                        indices.append(sym_to_int[base])
                    # 偶数次幂直接忽略 (变为 Identity)
            else:
                if factor in sym_to_int:
                    indices.append(sym_to_int[factor])

        # 步骤 B: 调用核心化简逻辑
        simplified_indices = reduce_observable_indices(indices)

        # 步骤 C: 转换回 SymPy 对象
        if not simplified_indices:
            final_terms.append(coeff)
        else:
            ops = [int_to_sym[idx] for idx in simplified_indices]
            final_terms.append(coeff * Mul(*ops))

    return Add(*final_terms)


# ==========================================
# 投影算子化简逻辑 (保持不变)
# ==========================================
def reduce_projector_indices(indices):
    """
    参考逻辑的具体实现。E0->0, E1->1, F0->2, F1->3
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

            # 规则: E^2 = E (合并)
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
                    indices.extend([idx] * int(exp))
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
# 简单的 Observables 预处理 (保持不变，用于rho_swap构建过程)
# ==========================================
def simplify_observables_basic(expr, ops_dict):
    A0, A1, B0, B1 = ops_dict['A0'], ops_dict['A1'], ops_dict['B0'], ops_dict['B1']
    subs_rules = {A0 ** 2: 1, A1 ** 2: 1, B0 ** 2: 1, B1 ** 2: 1}
    # 仅做最基本的平方消除，不做复杂排序
    return expand(expand(expr).subs(subs_rules))


# ==========================================
# 主程序
# ==========================================
def generate_fidelity_proof():
    # 1. 定义符号
    A0, A1 = symbols('A0 A1', commutative=False)
    B0, B1 = symbols('B0 B1', commutative=False)
    ops_dict = {'A0': A0, 'A1': A1, 'B0': B0, 'B1': B1}

    E0, E1 = symbols('E0 E1', commutative=False)
    F0, F1 = symbols('F0 F1', commutative=False)
    proj_dict = {'E0': E0, 'E1': E1, 'F0': F0, 'F1': F1}

    theta, mu = symbols('theta mu', real=True)
    Identity = 1

    print("=== 1. 正在构建 SWAP 算符 ===")

    vec_A = [
        (Identity + A0) / 2,
        A1 * (Identity - A0) / 2
    ]

    Z_B = (B0 + B1) / (2 * sympy.cos(mu))
    X_B = (B0 - B1) / (2 * sympy.sin(mu))

    vec_B = [
        (Identity + Z_B) / 2,
        X_B * (Identity - Z_B) / 2
    ]

    Omega = []
    for op_a in vec_A:
        for op_b in vec_B:
            Omega.append(op_a * op_b)

    print("=== 2. 显式输出 rho_swap 矩阵 (Observables 形式) ===")

    rho_swap = [[0] * 4 for _ in range(4)]
    for i in range(4):
        for j in range(4):
            raw_expr = Omega[i] * Omega[j]
            # 此时仅做基本化简，保留计算过程
            simplified_expr = simplify_observables_basic(raw_expr, ops_dict)
            rho_swap[i][j] = simplified_expr

    print("\n=== 3. 保真度方程 (Observables 形式) ===")

    c = sympy.cos(theta)
    s = sympy.sin(theta)

    F_op_raw = c ** 2 * rho_swap[0][0] + \
               c * s * rho_swap[0][3] + \
               s * c * rho_swap[3][0] + \
               s ** 2 * rho_swap[3][3]

    # --- 调用新的可观测量高级化简函数 ---
    print("正在进行 Observables 高级化简 (A^2=I & Sorting)...")
    F_op = simplify_observables_advanced(F_op_raw, ops_dict)
    print("F_op (Observables) 预览:")
    print(str(F_op)[:200] + "...")

    # --- 提取 Observables 系数并保存 JSON ---
    print("\n=== 3.1 正在提取 Observables 系数生成 JSON ===")
    terms_obs = Add.make_args(F_op)
    coeff_dict_obs = {}
    for term in terms_obs:
        c_factors, nc_factors = term.args_cnc()
        coeff = Mul(*c_factors)
        if not nc_factors:
            key = "I"
        else:
            ops_str = [str(f) for f in nc_factors]
            key = " ".join(ops_str)
            if not key: key = "I"

        if key in coeff_dict_obs:
            coeff_dict_obs[key] += coeff
        else:
            coeff_dict_obs[key] = coeff

    # 输出 Observables JSON
    final_json_dict_obs = {k: str(simplify(v)).replace('sin', 'np.sin').replace('cos', 'np.cos')
                           for k, v in coeff_dict_obs.items() if simplify(v) != 0}

    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        current_dir = os.getcwd()
    target_dir = os.path.join(current_dir, 'func')
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    file_path_obs = os.path.join(target_dir, 'fidelity_coeffs_obs.json')
    with open(file_path_obs, 'w') as f:
        f.write(json.dumps(final_json_dict_obs, indent=2))
    print(f"Observables JSON 文件已保存至: {file_path_obs}")

    print("\n=== 4. 正在转换为投影算子形式并展开 ===")

    subs_dict = {
        A0: 2 * E0 - 1,
        A1: 2 * E1 - 1,
        B0: 2 * F0 - 1,
        B1: 2 * F1 - 1
    }

    F_proj_raw = F_op.subs(subs_dict)

    print("\n=== 5. 调用参考逻辑进行化简 (Idempotency & Sorting) ===")
    F_proj = simplify_projectors_advanced(F_proj_raw, proj_dict)

    print("Projectors 化简结果预览:")
    print(str(F_proj)[:200] + "...")

    print("\n=== 6. 正在提取 Projectors 系数生成 JSON ===")

    terms = Add.make_args(F_proj)
    coeff_dict = {}

    for term in terms:
        c_factors, nc_factors = term.args_cnc()
        coeff = Mul(*c_factors)

        if not nc_factors:
            key = "I"
        else:
            # 此时 nc_factors 已经是排序且化简过的 (E在F前)
            ops_str = [str(f) for f in nc_factors]
            key = " ".join(ops_str)
            if not key: key = "I"

        if key in coeff_dict:
            coeff_dict[key] += coeff
        else:
            coeff_dict[key] = coeff

    # 输出 Projectors JSON
    final_json_dict = {k: str(simplify(v)).replace('sin', 'np.sin').replace('cos', 'np.cos')
                       for k, v in coeff_dict.items() if simplify(v) != 0}

    file_path = os.path.join(target_dir, 'fidelity_coeffs.json')

    with open(file_path, 'w') as f:
        f.write(json.dumps(final_json_dict, indent=2))

    print(f"\nProjectors JSON 文件已保存至: {file_path}")
    print("Projectors JSON 内容预览 (前 5 项):")
    count = 0
    for k, v in final_json_dict.items():
        print(f'  "{k}": "{v}"')
        count += 1
        if count >= 5: break
    print("  ...")


if __name__ == "__main__":
    generate_fidelity_proof()