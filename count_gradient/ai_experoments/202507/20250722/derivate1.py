import numpy as np
from sympy import symbols, sin, cos, asin, diff, sqrt, pi, atan, simplify

# 定义符号变量
theta = pi / 4
a0 = 0
a1 = pi / 2
b0 = pi / 4     # atan(sin(2 * theta))
b1 = 3 * pi /4  # b0  # b0 = b1

# 计算 A0, A1, B0, B1
A0 = cos(2 * theta) * cos(a0)
A1 = cos(2 * theta) * cos(a1)
B0 = cos(2 * theta) * cos(b0)
B1 = cos(2 * theta) * cos(b1)

# 计算 E00, E01, E10, E11
E00 = cos(a0) * cos(b0) + sin(2 * theta) * sin(a0) * sin(b0)
E01 = cos(a0) * cos(b1) + sin(2 * theta) * sin(a0) * sin(b1)
E10 = cos(a1) * cos(b0) + sin(2 * theta) * sin(a1) * sin(b0)
E11 = cos(a1) * cos(b1) + sin(2 * theta) * sin(a1) * sin(b1)  # 注意使用tilted的时候E11 的符号是减

# 打印计算值
print("A0:", A0.evalf())
print("A1:", A1.evalf())
print("B0:", B0.evalf())
print("B1:", B1.evalf())
print("E00:", E00.evalf())
print("E01:", E01.evalf())
print("E10:", E10.evalf())
print("E11:", E11.evalf())

# 定义四个等式
def compute_equation(s, t):
    u1 = (E00 + s * B0) / (1 + s * A0)
    u2 = (E10 + t * B0) / (1 + t * A1)
    u3 = (E01 + s * B1) / (1 + s * A0)
    u4 = (E11 + t * B1) / (1 + t * A1)
    eq = asin(u1) + asin(u2) + asin(u3) - asin(u4) - pi
    return eq.evalf()

# 计算四个等式
f11 = compute_equation(1, 1)   # s=1, t=1
f10 = compute_equation(1, -1)  # s=1, t=-1
f01 = compute_equation(-1, 1)  # s=-1, t=1
f00 = compute_equation(-1, -1) # s=-1, t=-1

print("\nEquations:")
print("f11:", f11)
print("f10:", f10)
print("f01:", f01)
print("f00:", f00)

# 定义符号变量
A0_sym, B0_sym, A1_sym, B1_sym, E00_sym, E01_sym, E10_sym, E11_sym, s_sym, t_sym = symbols('A0 B0 A1 B1 E00 E01 E10 E11 s t')

# 计算偏导数
def compute_partial(s_val, t_val):
    u1 = (E00_sym + s_val * B0_sym) / (1 + s_val * A0_sym)
    u2 = (E10_sym + t_val * B0_sym) / (1 + t_val * A1_sym)
    u3 = (E01_sym + s_val * B1_sym) / (1 + s_val * A0_sym)
    u4 = (E11_sym + t_val * B1_sym) / (1 + t_val * A1_sym)
    F = asin(u1) + asin(u2) + asin(u3) - asin(u4) - pi

    partials = {
        'A0': diff(F, A0_sym),
        'B0': diff(F, B0_sym),
        'A1': diff(F, A1_sym),
        'B1': diff(F, B1_sym),
        'E00': diff(F, E00_sym),
        'E01': diff(F, E01_sym),
        'E10': diff(F, E10_sym),
        'E11': diff(F, E11_sym),
    }
    return partials

# 计算四种情况的偏导数
partials11 = compute_partial(1, 1)   # s=1, t=1
partials10 = compute_partial(1, -1)  # s=1, t=-1
partials01 = compute_partial(-1, 1)  # s=-1, t=1
partials00 = compute_partial(-1, -1) # s=-1, t=-1

# 定义替换字典
subs_dict = {
    A0_sym: A0,
    A1_sym: A1,
    B0_sym: B0,
    B1_sym: B1,
    E00_sym: E00,
    E01_sym: E01,
    E10_sym: E10,
    E11_sym: E11,
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

# 打印所有偏导数
print("\nPartial derivatives:")
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