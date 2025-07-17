from sympy import symbols, sin, asin, diff, pi, cos, sqrt, Eq, solve, simplify

# 定义变量
x0, x1, x2, x3, x4, x5, x6, x7 = symbols('x0 x1 x2 x3 x4 x5 x6 x7')
A0, A1, B0, B1, E00, E10, E11 = symbols('A0 A1 B0 B1 E00 E10 E11')

# 给定的切平面方程（已简化）
tangent_plane_eq = Eq(
    -B1 - x7
    + (-1 - (A0 + 1)*cos(asin((B0 + E00)/(A0 + 1)) + asin((B0 + E10)/(A1 + 1)) + asin((B1 + E11)/(A1 + 1)))/(sqrt(1 - (B1 + E11)**2/(A1 + 1)**2)*(A1 + 1)))*(-B1 + x3)
    + (-A0 + x0)*(-sin(asin((B0 + E00)/(A0 + 1)) + asin((B0 + E10)/(A1 + 1)) + asin((B1 + E11)/(A1 + 1)))
    + (B0 + E00)*cos(asin((B0 + E00)/(A0 + 1)) + asin((B0 + E10)/(A1 + 1)) + asin((B1 + E11)/(A1 + 1)))/(sqrt(1 - (B0 + E00)**2/(A0 + 1)**2)*(A0 + 1)))
    - (A0 + 1)*(-A1 + x1)*(-(B1 + E11)/(sqrt(1 - (B1 + E11)**2/(A1 + 1)**2)*(A1 + 1)**2)
    - (B0 + E10)/(sqrt(1 - (B0 + E10)**2/(A1 + 1)**2)*(A1 + 1)**2))*cos(asin((B0 + E00)/(A0 + 1)) + asin((B0 + E10)/(A1 + 1)) + asin((B1 + E11)/(A1 + 1)))
    - (A0 + 1)*(-B0 + x2)*(1/(sqrt(1 - (B0 + E10)**2/(A1 + 1)**2)*(A1 + 1))
    + 1/(sqrt(1 - (B0 + E00)**2/(A0 + 1)**2)*(A0 + 1)))*cos(asin((B0 + E00)/(A0 + 1)) + asin((B0 + E10)/(A1 + 1)) + asin((B1 + E11)/(A1 + 1)))
    - (A0 + 1)*sin(asin((B0 + E00)/(A0 + 1)) + asin((B0 + E10)/(A1 + 1)) + asin((B1 + E11)/(A1 + 1)))
    - (A0 + 1)*(-E11 + x6)*cos(asin((B0 + E00)/(A0 + 1)) + asin((B0 + E10)/(A1 + 1)) + asin((B1 + E11)/(A1 + 1)))/(sqrt(1 - (B1 + E11)**2/(A1 + 1)**2)*(A1 + 1))
    - (A0 + 1)*(-E10 + x5)*cos(asin((B0 + E00)/(A0 + 1)) + asin((B0 + E10)/(A1 + 1)) + asin((B1 + E11)/(A1 + 1)))/(sqrt(1 - (B0 + E10)**2/(A1 + 1)**2)*(A1 + 1))
    - (-E00 + x4)*cos(asin((B0 + E00)/(A0 + 1)) + asin((B0 + E10)/(A1 + 1)) + asin((B1 + E11)/(A1 + 1)))/sqrt(1 - (B0 + E00)**2/(A0 + 1)**2),
    0
)

# 提取系数和常数项（假设已整理为线性形式）
# 这里需要手动或通过符号计算提取 x_i 的系数
# 示例：假设方程为 c0*(x0 - A0) + c1*(x1 - A1) + ... + c7*(x7 - E01) = 0
# 可以解出 x7 作为其他变量的函数

# 选择自由变量（例如 x0, x1, ..., x6），解出 x7
free_vars = [x0, x1, x2, x3, x4, x5, x6]
x7_expr = solve(tangent_plane_eq, x7)[0]  # 解出 x7 的表达式

# 生成切平面上的向量：
# 方法：固定某些自由变量为 1，其余为 0，计算 x7 的值
def get_tangent_vector(free_var_index):
    # 设置自由变量：第 free_var_index 个为 1，其余为 0
    substitutions = {var: (1 if i == free_var_index else 0) for i, var in enumerate(free_vars)}
    x7_val = x7_expr.subs(substitutions)
    vector = [substitutions.get(var, 0) for var in free_vars] + [x7_val]
    return vector

# 示例：生成两个线性无关的向量
vector1 = get_tangent_vector(0)  # x0=1, x1=0, ..., x6=0
vector2 = get_tangent_vector(1)  # x0=0, x1=1, ..., x6=0

print("切平面上的向量 1:", vector1)
print("切平面上的向量 2:", vector2)