import numpy as np

# 给定参数值
A0 = 0.37796447
A1 = 0.37796447
B0 = 0.5
B1 = 0.0
E00 = 0.75592895
E01 = -0.56694671
E10 = 0.75592895
E11 = 0.56694671

# 常数 s 和 t（根据题目要求，保留符号形式，但计算时需要具体值）
# 这里假设 s 和 t 为1，如果需要其他值请修改
s = 1.0
t = -1.0

# 计算中间变量 X
def calculate_X(E, A, B, alpha):
    """计算 X = (E + alpha * B) / (1 + alpha * A)"""
    return (E + alpha * B) / (1 + alpha * A)

# 计算各个 X 值
X00 = calculate_X(E00, A0, B0, s)
X01 = calculate_X(E01, A0, B1, s)
X10 = calculate_X(E10, A1, B0, t)
X11 = calculate_X(E11, A1, B1, t)

print(f"X00 = {X00}")
print(f"X01 = {X01}")
print(f"X10 = {X10}")
print(f"X11 = {X11}")

# 计算平方根项
sqrt_1_minus_X00_sq = np.sqrt(1 - X00**2)
sqrt_1_minus_X01_sq = np.sqrt(1 - X01**2)
sqrt_1_minus_X10_sq = np.sqrt(1 - X10**2)
sqrt_1_minus_X11_sq = np.sqrt(1 - X11**2)

# 计算分母项
denom_sA0 = 1 + s * A0
denom_tA1 = 1 + t * A1

# 计算各个偏导数
dF_dE00 = 1 / (denom_sA0 * sqrt_1_minus_X00_sq)
dF_dE01 = -1 / (denom_sA0 * sqrt_1_minus_X01_sq)
dF_dE10 = 1 / (denom_tA1 * sqrt_1_minus_X10_sq)
dF_dE11 = 1 / (denom_tA1 * sqrt_1_minus_X11_sq)

dF_dA0 = -s / denom_sA0 * (X00 / sqrt_1_minus_X00_sq - X01 / sqrt_1_minus_X01_sq)
dF_dA1 = -t / denom_tA1 * (X10 / sqrt_1_minus_X10_sq + X11 / sqrt_1_minus_X11_sq)

dF_dB0 = s / (denom_sA0 * sqrt_1_minus_X00_sq) + t / (denom_tA1 * sqrt_1_minus_X10_sq)
dF_dB1 = -s / (denom_sA0 * sqrt_1_minus_X01_sq) + t / (denom_tA1 * sqrt_1_minus_X11_sq)

# 输出结果
print("\n偏导数值:")
print(f"∂F/∂A0  = {dF_dA0}")
print(f"∂F/∂B0  = {dF_dB0}")
print(f"∂F/∂A1  = {dF_dA1}")
print(f"∂F/∂B1  = {dF_dB1}")
print(f"∂F/∂E00 = {dF_dE00}")
print(f"∂F/∂E01 = {dF_dE01}")
print(f"∂F/∂E10 = {dF_dE10}")
print(f"∂F/∂E11 = {dF_dE11}")


# 验证原方程是否成立（应该接近π）
F_value = np.arcsin(X00) + np.arcsin(X10) - np.arcsin(X01) + np.arcsin(X11)
print(f"\n原方程值 F = {F_value} (应该接近0)")

# 计算乘积求和
product_sum = (dF_dA0 * A0 + dF_dA1 * A1 +
               dF_dB0 * B0 + dF_dB1 * B1 +
               dF_dE00 * E00 + dF_dE01 * E01 +
               dF_dE10 * E10 + dF_dE11 * E11)

print(f"\n乘积求和结果:")
print(f"dF_dA0 * A0 + dF_dA1 * A1 + dF_dB0 * B0 + dF_dB1 * B1 +")
print(f"dF_dE00 * E00 + dF_dE01 * E01 + dF_dE10 * E10 + dF_dE11 * E11 = {product_sum}")