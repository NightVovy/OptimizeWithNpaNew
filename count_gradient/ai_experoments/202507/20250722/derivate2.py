import numpy as np
from sympy import symbols, sin, cos, asin, diff, sqrt, pi, atan, simplify

# 定义符号变量
theta_sym, a0_sym, a1_sym, b0_sym, b1_sym = symbols('theta a0 a1 b0 b1')
theta_val = pi / 6
a0_val = 0
a1_val = pi / 2
b0_val = atan(sin(2 * theta_val))
b1_val = b0_val  # b0 = b1

# 计算 A0, A1, B0, B1
A0 = cos(2 * theta_sym) * cos(a0_sym)
A1 = cos(2 * theta_sym) * cos(a1_sym)
B0 = cos(2 * theta_sym) * cos(b0_sym)
B1 = cos(2 * theta_sym) * cos(b1_sym)

# 计算 E00, E01, E10, E11
E00 = cos(a0_sym) * cos(b0_sym) + sin(2 * theta_sym) * sin(a0_sym) * sin(b0_sym)
E01 = cos(a0_sym) * cos(b1_sym) + sin(2 * theta_sym) * sin(a0_sym) * sin(b1_sym)
E10 = cos(a1_sym) * cos(b0_sym) + sin(2 * theta_sym) * sin(a1_sym) * sin(b0_sym)
E11 = cos(a1_sym) * cos(b1_sym) - sin(2 * theta_sym) * sin(a1_sym) * sin(b1_sym)

# 定义四个等式
def compute_equation(s, t):
    u1 = (E00 + s * B0) / (1 + s * A0)
    u2 = (E10 + t * B0) / (1 + t * A1)
    u3 = (E01 + s * B1) / (1 + s * A0)
    u4 = (E11 + t * B1) / (1 + t * A1)
    eq = asin(u1) + asin(u2) + asin(u3) - asin(u4) - pi
    return eq

# 计算四个等式
f11 = compute_equation(1, 1)   # s=1, t=1
f10 = compute_equation(1, -1)  # s=1, t=-1
f01 = compute_equation(-1, 1)  # s=-1, t=1
f00 = compute_equation(-1, -1) # s=-1, t=-1

# 计算偏导数
def compute_partial(f):
    partials = {
        'theta': diff(f, theta_sym),
        'a0': diff(f, a0_sym),
        'a1': diff(f, a1_sym),
        'b0': diff(f, b0_sym),
        'b1': diff(f, b1_sym),
    }
    return partials

# 计算四种情况的偏导数
partials11 = compute_partial(f11)
partials10 = compute_partial(f10)
partials01 = compute_partial(f01)
partials00 = compute_partial(f00)

# 定义替换字典
subs_dict = {
    theta_sym: theta_val,
    a0_sym: a0_val,
    a1_sym: a1_val,
    b0_sym: b0_val,
    b1_sym: b1_val,
}

# 计算偏导数的具体值
def evaluate_partials(partials):
    evaluated = {}
    for var, expr in partials.items():
        evaluated[var] = expr.subs(subs_dict).evalf()
    return evaluated

# 计算具体值
f11_partials = evaluate_partials(partials11)
f10_partials = evaluate_partials(partials10)
f01_partials = evaluate_partials(partials01)
f00_partials = evaluate_partials(partials00)

# 打印方程值
print("Equations evaluated at current point:")
print("f11:", f11.subs(subs_dict).evalf())
print("f10:", f10.subs(subs_dict).evalf())
print("f01:", f01.subs(subs_dict).evalf())
print("f00:", f00.subs(subs_dict).evalf())

# 打印法向量
print("\nNormal vectors (partial derivatives):")
print("f11 partials:")
for var, val in f11_partials.items():
    print(f"f11_{var}:", val)

print("\nf10 partials:")
for var, val in f10_partials.items():
    print(f"f10_{var}:", val)

print("\nf01 partials:")
for var, val in f01_partials.items():
    print(f"f01_{var}:", val)

print("\nf00 partials:")
for var, val in f00_partials.items():
    print(f"f00_{var}:", val)