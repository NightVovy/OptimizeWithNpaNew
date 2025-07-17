import numpy as np

# 定义贝尔不等式系数
coefficients = {
    'E0x': -0.30805177,
    'E1x': 0.016737746,
    'Ex0': -0.24839279,
    'Ex1': -0.28919283,
    'E00': -0.55811724,
    'E01': -0.5610094,
    'E10': -0.72157791, # ?
    'E11': -0.73465683
}

# 生成所有可能的经典策略 (A0, A1, B0, B1) ∈ {-1, +1}^4
strategies = [(a0, a1, b0, b1)
              for a0 in [-1, 1]
              for a1 in [-1, 1]
              for b0 in [-1, 1]
              for b1 in [-1, 1]]

# 计算每种策略的 Bell 表达式值
def compute_bell_value(a0, a1, b0, b1):
    # 单边期望值
    E0x = a0
    E1x = a1
    Ex0 = b0
    Ex1 = b1
    # 联合期望值
    E00 = a0 * b0
    E01 = a0 * b1
    E10 = a1 * b0
    E11 = a1 * b1
    # 计算 Bell 表达式
    value = (coefficients['E0x'] * E0x +
             coefficients['E1x'] * E1x +
             coefficients['Ex0'] * Ex0 +
             coefficients['Ex1'] * Ex1 +
             coefficients['E00'] * E00 +
             coefficients['E01'] * E01 +
             coefficients['E10'] * E10 +
             coefficients['E11'] * E11)
    return value

# 计算所有策略并找到最大值
max_value = -np.inf
best_strategy = None

for strategy in strategies:
    a0, a1, b0, b1 = strategy
    value = compute_bell_value(a0, a1, b0, b1)
    if value > max_value:
        max_value = value
        best_strategy = strategy

# 输出结果
print("所有策略的 Bell 值:")
for i, strategy in enumerate(strategies):
    a0, a1, b0, b1 = strategy
    value = compute_bell_value(a0, a1, b0, b1)
    print(f"策略 {i+1}: (A0={a0}, A1={a1}, B0={b0}, B1={b1}) → I' = {value:.6f}")

print("\n经典最大值:")
print(f"策略 (A0={best_strategy[0]}, A1={best_strategy[1]}, B0={best_strategy[2]}, B1={best_strategy[3]})")
print(f"最大值 I' = {max_value:.6f}")