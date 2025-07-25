from sympy import symbols, sin, cos, asin, pi, atan, diff, Matrix, simplify

# 定义符号变量
theta_sym, a0_sym, a1_sym, b0_sym, b1_sym = symbols('theta a0 a1 b0 b1')


# 定义计算方程的函数（完全展开形式）
def compute_equation(s, t, theta, a0, a1, b0, b1):
    # 计算各项
    u1 = ((cos(a0) * cos(b0) + sin(2 * theta) * sin(a0) * sin(b0) + s * cos(2 * theta) * cos(b0)) /
          (1 + s * cos(2 * theta) * cos(a0)))

    u2 = ((cos(a1) * cos(b0) + sin(2 * theta) * sin(a1) * sin(b0) + t * cos(2 * theta) * cos(b0)) /
          (1 + t * cos(2 * theta) * cos(a1)))

    u3 = ((cos(a0) * cos(b1) + sin(2 * theta) * sin(a0) * sin(b1) + s * cos(2 * theta) * cos(b1)) /
          (1 + s * cos(2 * theta) * cos(a0)))

    u4 = ((cos(a1) * cos(b1) - sin(2 * theta) * sin(a1) * sin(b1) + t * cos(2 * theta) * cos(b1)) /
          (1 + t * cos(2 * theta) * cos(a1)))

    eq = asin(u1) + asin(u2) + asin(u3) - asin(u4) - pi
    return eq


# 设置参数值
theta_val = pi / 6
a0_val = 0
a1_val = pi / 2
b0_val = atan(sin(2 * theta_val))
b1_val = atan(sin(2 * theta_val))

# 计算四个方程的值
f11 = compute_equation(1, 1, theta_val, a0_val, a1_val, b0_val, b1_val)
f10 = compute_equation(1, -1, theta_val, a0_val, a1_val, b0_val, b1_val)
f01 = compute_equation(-1, 1, theta_val, a0_val, a1_val, b0_val, b1_val)
f00 = compute_equation(-1, -1, theta_val, a0_val, a1_val, b0_val, b1_val)

print("f11的值:", f11.evalf())
print("f10的值:", f10.evalf())
print("f01的值:", f01.evalf())
print("f00的值:", f00.evalf())


# 定义符号表达式以计算法向量
def compute_normal_vector(s, t):
    eq = compute_equation(s, t, theta_sym, a0_sym, a1_sym, b0_sym, b1_sym)
    grad = [diff(eq, var) for var in [theta_sym, a0_sym, a1_sym, b0_sym, b1_sym]]
    return grad


# 计算四个方程的法向量
normal_f11 = compute_normal_vector(1, 1)
normal_f10 = compute_normal_vector(1, -1)
normal_f01 = compute_normal_vector(-1, 1)
normal_f00 = compute_normal_vector(-1, -1)

# 在给定点处求值
subs_dict = {
    theta_sym: theta_val,
    a0_sym: a0_val,
    a1_sym: a1_val,
    b0_sym: b0_val,
    b1_sym: b1_val
}


# 计算数值法向量
def eval_normal(normal):
    return [component.subs(subs_dict).evalf() for component in normal]


normal_f11_val = eval_normal(normal_f11)
normal_f10_val = eval_normal(normal_f10)
normal_f01_val = eval_normal(normal_f01)
normal_f00_val = eval_normal(normal_f00)

# 打印结果
print("\n法向量 (对theta, a0, a1, b0, b1的偏导):")
print("f11的法向量:", [x for x in normal_f11_val])
print("f10的法向量:", [x for x in normal_f10_val])
print("f01的法向量:", [x for x in normal_f01_val])
print("f00的法向量:", [x for x in normal_f00_val])