import math


def trigonometric_calculator():
    print("三角函数计算器（预设值版本）")

    # 预设值
    a0 = 0  # 弧度值
    a1 = math.pi / 2  # 弧度值
    b0 = math.pi / 4  # 弧度值
    b1 = 3 * math.pi / 4  # π弧度（约3.1416）
    x1 =  math.pi / 4
    x2 =  3 * math.pi / 4
    theta = math.pi / 6

    # 计算sin值
    sin_a0 = math.sin(a0)
    sin_a1 = math.sin(a1)
    sin_b0 = math.sin(b0)
    sin_b1 = math.sin(b1)

    # 计算cos值
    cos_a0 = math.cos(a0)
    cos_a1 = math.cos(a1)
    cos_b0 = math.cos(b0)
    cos_b1 = math.cos(b1)

    # 计算cot玩玩
    # 计算tan值
    tan_x1 = math.tan(x1)
    tan_x2 = math.tan(x2)


    # 计算cot值（注意处理tan为0的情况）
    try:
        cot_x1 = 1 / tan_x1
    except ZeroDivisionError:
        cot_x1 = float('inf')  # 当tan(x1)为0时，cot(x1)为无穷大

    try:
        cot_x2 = 1 / tan_x2
    except ZeroDivisionError:
        cot_x2 = float('inf')  # 当tan(x2)为0时，cot(x2)为无穷大


    # 计算公式值，此处为alpha不等式，即alphaA0
    # 计算mu值（新增部分）
    mu = math.atan(math.sin(2 * theta))
    alpha = 2 / math.sqrt(1 + 2 * (math.tan(2 * theta))**2)
    # 计算sin(mu)和cos(mu)（新增部分）
    sin_mu = math.sin(mu)
    cos_mu = math.cos(mu)

    # 显示预设值和计算结果
    print("\n预设值:")
    print(f"a0 = {a0:.4f} 弧度, a1 = {a1:.4f} 弧度")
    print(f"b0 = {b0:.4f} 弧度, b1 = π ≈ {b1:.4f} 弧度")
    print(f"x1 = {x1:.4f} 弧度, x2 = {x2:.4f} 弧度")
    print(f"theta = {theta:.4f} 弧度 (约{math.degrees(theta):.1f}度)")  # 新增theta值显示

    print("\n计算结果:")
    print(f" cos(a0) = {cos_a0:.4f}, sin(a0) = {sin_a0:.4f}")
    print(f" cos(a1) = {cos_a1:.4f}, sin(a1) = {sin_a1:.4f}")
    print(f" cos(b0) = {cos_b0:.4f}, sin(b0) = {sin_b0:.4f}")
    print(f" cos(b1) = {cos_b1:.4f}, sin(b1) = {sin_b1:.4f}")
    print(f" cot(x1) = {cot_x1:.4f}, cot(x2) = {cot_x2:.4f}")
    print(f" mu = {mu:.4f} 弧度 (约{math.degrees(mu):.1f}度)")  # 新增mu值显示
    print(f" sin(mu) = {sin_mu:.4f}, cos(mu) = {cos_mu:.4f}")  # 新增sin(mu)和cos(mu)显示
    print(f" alpha = {alpha:.4f}")


# 运行计算器
if __name__ == "__main__":
    trigonometric_calculator()