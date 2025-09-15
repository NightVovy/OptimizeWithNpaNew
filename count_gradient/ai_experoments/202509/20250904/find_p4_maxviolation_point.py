import numpy as np
from scipy.optimize import minimize

# 给定的法向量 n3（字符串列表转换为浮点数）
n3 = np.array(['1.17588953', '1.17588951', '0.00000012', '0.00000019',
               '1.76383428', '-2.35177882', '1.76383427', '2.35177900'], dtype=float)

# 最大量子违背值（需要你提供）
max_q_violation = 7.5294

def objective(params):
    theta, a0, a1, b0, b1 = params
    # 计算P4的各个分量
    cos2t = np.cos(2 * theta)
    sin2t = np.sin(2 * theta)
    A0 = cos2t * np.cos(a0)
    A1 = cos2t * np.cos(a1)
    B0 = cos2t * np.cos(b0)
    B1 = cos2t * np.cos(b1)
    E00 = np.cos(a0)*np.cos(b0) + sin2t * np.sin(a0)*np.sin(b0)
    E01 = np.cos(a0)*np.cos(b1) + sin2t * np.sin(a0)*np.sin(b1)
    E10 = np.cos(a1)*np.cos(b0) + sin2t * np.sin(a1)*np.sin(b0)
    E11 = np.cos(a1)*np.cos(b1) + sin2t * np.sin(a1)*np.sin(b1)
    P4 = np.array([A0, A1, B0, B1, E00, E01, E10, E11])
    # 计算n3与P4的点积
    dot_product = np.dot(n3, P4)
    # 我们希望最大化点积，因此最小化负点积
    return -dot_product

# 参数边界：theta在(0, pi/4], a0,a1,b0,b1在(0, pi)
bounds = [
    (1e-6, np.pi/4),   # theta 避免0，因为0可能无意义
    (1e-6, np.pi-1e-6), # a0
    (1e-6, np.pi-1e-6), # a1
    (1e-6, np.pi-1e-6), # b0
    (1e-6, np.pi-1e-6)  # b1
]

# 初始猜测（随机选择或根据经验）
initial_guess = [np.pi/4, np.pi/2, np.pi/2, np.pi/2, np.pi/2]

# 进行优化
result = minimize(objective, initial_guess, bounds=bounds, method='L-BFGS-B')

# 提取最优参数
theta_opt, a0_opt, a1_opt, b0_opt, b1_opt = result.x
max_dot = -result.fun  # 因为最小化了负点积

# 检查是否达到max_q_violation（允许一定误差）
tolerance = 1e-4
if abs(max_dot - max_q_violation) < tolerance:
    print("成功找到达到最大量子违背值的参数：")
else:
    print("未完全达到最大量子违背值，但找到了当前最优参数：")

print(f"最优 theta: {theta_opt} 弧度")
print(f"最优 a0: {a0_opt} 弧度")
print(f"最优 a1: {a1_opt} 弧度")
print(f"最优 b0: {b0_opt} 弧度")
print(f"最优 b1: {b1_opt} 弧度")
print(f"最大点积值: {max_dot}")
print(f"目标最大量子违背值: {max_q_violation}")

# 计算最优参数下的P4点
theta = theta_opt
a0 = a0_opt
a1 = a1_opt
b0 = b0_opt
b1 = b1_opt
cos2t = np.cos(2 * theta)
sin2t = np.sin(2 * theta)
A0 = cos2t * np.cos(a0)
A1 = cos2t * np.cos(a1)
B0 = cos2t * np.cos(b0)
B1 = cos2t * np.cos(b1)
E00 = np.cos(a0)*np.cos(b0) + sin2t * np.sin(a0)*np.sin(b0)
E01 = np.cos(a0)*np.cos(b1) + sin2t * np.sin(a0)*np.sin(b1)
E10 = np.cos(a1)*np.cos(b0) + sin2t * np.sin(a1)*np.sin(b0)
E11 = np.cos(a1)*np.cos(b1) + sin2t * np.sin(a1)*np.sin(b1)
P4_opt = [A0, A1, B0, B1, E00, E01, E10, E11]

print("最优P4点坐标：")
print(P4_opt)