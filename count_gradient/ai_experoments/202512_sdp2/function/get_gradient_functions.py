import sympy as sp


def generate_partial_derivatives():
    # 1. 定义符号
    # s, t: 参数
    # A0, A1, B0, B1: 边缘项系数
    # E00, E01, E10, E11: 相关项系数
    s, t = sp.symbols('s t', real=True)
    A0, A1, B0, B1 = sp.symbols('A0 A1 B0 B1', real=True)
    E00, E01, E10, E11 = sp.symbols('E00 E01 E10 E11', real=True)

    # 将系数打包以便后续遍历求导
    coeffs = {
        'A0': A0, 'A1': A1,
        'B0': B0, 'B1': B1,
        'E00': E00, 'E01': E01,
        'E10': E10, 'E11': E11
    }

    # 2. 定义辅助函数 (对应原代码中的 compute_arcsin_term)
    # 注意：符号计算不需要 np.clip，我们假设在定义域内
    def get_term(Ax, alpha, By, s_or_t, Exy):
        numerator = Exy + s_or_t * By
        denominator = 1 + s_or_t * Ax
        return sp.asin(numerator / denominator)

    # 3. 构建总约束方程 (对应 constraint_equation)
    # 公式: term1 + term2 + term3 - term4 - pi
    term1 = get_term(A0, s, B0, s, E00)
    term2 = get_term(A1, t, B0, t, E10)
    term3 = get_term(A0, s, B1, s, E01)
    term4 = get_term(A1, t, B1, t, E11)

    equation = term1 + term2 + term3 - term4 - sp.pi

    print("=== 目标函数 F 的定义 ===")
    print(f"F = {term1} + {term2} + {term3} - {term4} - pi\n")
    print("=== 8个系数的偏导数公式 (Gradient) ===\n")

    # 4. 对每个系数求偏导并输出
    for name, symbol in coeffs.items():
        # 求偏导
        derivative = sp.diff(equation, symbol)
        # 简化公式 (可选，有时简化后的形式更易读)
        simplified_derivative = sp.simplify(derivative)

        print(f"--- ∂F / ∂{name} ---")
        # 打印纯文本形式
        print(simplified_derivative)
        print("\n")


if __name__ == "__main__":
    generate_partial_derivatives()