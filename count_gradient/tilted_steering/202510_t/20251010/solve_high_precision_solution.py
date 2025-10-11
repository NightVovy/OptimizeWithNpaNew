import math
import random
from scipy.optimize import minimize


def calculate_lambda_2(p00, p01, p10, p11, beta1, beta2, mu1, mu2, theta, alpha):
    part1 = alpha * math.cos(beta1)
    part2 = (p10 * math.sin(mu1) - p11 * math.sin(mu2)) * math.sin(beta2) * (math.sin(theta) / math.cos(theta))
    part3 = (p00 * math.sin(mu1) + p01 * math.sin(mu2)) * math.sin(beta1) * (math.sin(theta) / math.cos(theta))
    return part1 + part2 + part3


def calculate_lambda_3(p00, p01, p10, p11, beta1, beta2, mu1, mu2, theta, alpha):
    part1 = -alpha * math.cos(beta1)
    part2 = (p10 * math.sin(mu1) - p11 * math.sin(mu2)) * math.sin(beta2) * (math.cos(theta) / math.sin(theta))
    part3 = (p00 * math.sin(mu1) + p01 * math.sin(mu2)) * math.sin(beta1) * (math.cos(theta) / math.sin(theta))
    return part1 + part2 + part3


def equation_1(p00, p01, p10, p11, beta1, beta2, mu1, mu2, alpha):
    part1 = alpha * math.sin(beta1)
    part2 = (p00 * math.cos(beta1) + p10 * math.cos(beta2)) * math.sin(mu1)
    part3 = (p01 * math.cos(beta1) - p11 * math.cos(beta2)) * math.sin(mu2)
    return part1 + part2 + part3


def equation_2(p00, p01, p10, p11, beta1, beta2, mu1, mu2):
    part1 = (p00 * math.sin(beta1) + p10 * math.sin(beta2)) * math.cos(mu1)
    part2 = (p01 * math.sin(beta1) - p11 * math.sin(beta2)) * math.cos(mu2)
    return part1 + part2


def objective(params):
    # 参数中只包含需要优化的变量
    mu1, mu2, alpha, theta = params
    # 固定参数值
    p = 0.85
    q = 0.6
    beta1 = 0.0
    beta2 = math.pi / 2  # 固定为π/2

    # 计算p系数
    p00 = p * q
    p01 = p * (1 - q)
    p10 = (1 - p) * q
    p11 = (1 - p) * (1 - q)

    # 计算核心变量
    lambda2 = calculate_lambda_2(p00, p01, p10, p11, beta1, beta2, mu1, mu2, theta, alpha)
    lambda3 = calculate_lambda_3(p00, p01, p10, p11, beta1, beta2, mu1, mu2, theta, alpha)
    eq1 = equation_1(p00, p01, p10, p11, beta1, beta2, mu1, mu2, alpha)
    eq2 = equation_2(p00, p01, p10, p11, beta1, beta2, mu1, mu2)

    # 惩罚mu1和mu2接近0的情况
    mu1_penalty = 1000.0 / (mu1 ** 2 + 1e-6)  # 当mu1接近0时惩罚增大
    mu2_penalty = 1000.0 / (mu2 ** 2 + 1e-6)  # 当mu2接近0时惩罚增大

    # 角度为0的数量检查（只有beta1固定为0）
    zero_angles = 1  # beta1固定为0，mu1和mu2不允许为0
    zero_angle_penalty = 10000.0 if zero_angles > 1 else 0.0

    # 核心条件惩罚
    lambda_diff_penalty = 1000 * (lambda2 - lambda3) ** 2  # 确保lambda2=lambda3
    eq1_penalty = 1000 * eq1 ** 2  # 确保eq1=0
    eq2_penalty = 1000 * eq2 ** 2  # 确保eq2=0

    # 确保lambda不为0
    lambda_magnitude = (lambda2 ** 2 + lambda3 ** 2) / 2
    lambda_zero_penalty = 100.0 / (lambda_magnitude + 1e-6) if lambda_magnitude < 0.1 else 0.0

    # alpha惩罚（确保不接近0）
    alpha_penalty = 100.0 / alpha ** 2 if alpha < 0.5 else 0.0

    # 总目标函数
    total = (lambda_diff_penalty + eq1_penalty + eq2_penalty +
             lambda_zero_penalty + alpha_penalty +
             zero_angle_penalty + mu1_penalty + mu2_penalty)

    return total


def generate_random_guess():
    """生成随机的初始猜测值，确保mu1和mu2远离0"""
    mu1 = random.uniform(0.1, math.pi)  # 远离0
    mu2 = random.uniform(0.1, math.pi)  # 远离0
    alpha = random.uniform(0.5, 10)
    theta = random.uniform(0.01, math.pi / 4 - 0.01)
    return [mu1, mu2, alpha, theta]


# 固定参数值
p = 0.85
q = 0.6
beta1 = 0.0
beta2 = math.pi / 2

# 计算固定p和q对应的p系数
p00 = p * q
p01 = p * (1 - q)
p10 = (1 - p) * q
p11 = (1 - p) * (1 - q)

print(f"固定参数值：")
print(f"p = {p}, q = {q}")
print(f"beta1 = {beta1} 弧度 ({math.degrees(beta1):.2f} 度)")
print(f"beta2 = {beta2:.6f} 弧度 ({math.degrees(beta2):.2f} 度)")
print(f"计算得到的p系数：")
print(f"p00 = {p00:.6f}, p01 = {p01:.6f}")
print(f"p10 = {p10:.6f}, p11 = {p11:.6f}\n")

# 参数边界（确保mu1和mu2远离0）
bounds = [
    (0.01, math.pi),  # mu1（不允许为0）
    (0.01, math.pi),  # mu2（不允许为0）
    (0.5, 10),  # alpha (>0且不接近0)
    (0.01, math.pi / 4 - 0.01)  # theta
]

# 优化参数设置
optimization_kwargs = {
    'method': 'L-BFGS-B',
    'bounds': bounds,
    'tol': 1e-12,
    'options': {
        'maxiter': 10000,
        'gtol': 1e-10
    }
}

# 持续尝试直到找到满足所有条件的解
attempt = 0
best_result = None
best_error = float('inf')
required_precision = 1e-6  # 要求的精度
max_attempts = 1000  # 最大尝试次数

print(f"开始寻找满足条件的解，要求精度: {required_precision}")
print(f"将尝试最多 {max_attempts} 次...\n")

while attempt < max_attempts:
    attempt += 1
    if attempt % 10 == 0:
        print(f"已尝试 {attempt} 次，最佳误差: {best_error:.6e}")

    # 生成随机初始猜测值
    guess = generate_random_guess()

    # 执行优化
    result = minimize(objective, guess, **optimization_kwargs)

    if result.success:
        # 计算验证值
        mu1, mu2, alpha, theta = result.x

        lambda2 = calculate_lambda_2(p00, p01, p10, p11, beta1, beta2, mu1, mu2, theta, alpha)
        lambda3 = calculate_lambda_3(p00, p01, p10, p11, beta1, beta2, mu1, mu2, theta, alpha)
        eq1 = equation_1(p00, p01, p10, p11, beta1, beta2, mu1, mu2, alpha)
        eq2 = equation_2(p00, p01, p10, p11, beta1, beta2, mu1, mu2)

        # 计算总误差
        total_error = abs(lambda2 - lambda3) + abs(eq1) + abs(eq2)

        # 检查是否是更好的解
        if total_error < best_error:
            best_error = total_error
            best_result = (p, q, beta1, beta2, mu1, mu2, alpha, theta,
                           lambda2, lambda3, eq1, eq2)

        # 检查是否满足所有条件
        lambda_magnitude = (lambda2 ** 2 + lambda3 ** 2) / 2
        mu1_ok = mu1 > 1e-6  # 确保mu1不为0
        mu2_ok = mu2 > 1e-6  # 确保mu2不为0

        conditions_met = (abs(lambda2 - lambda3) < required_precision and
                          abs(eq1) < required_precision and
                          abs(eq2) < required_precision and
                          alpha > 0.5 and
                          lambda_magnitude > required_precision and
                          mu1_ok and mu2_ok)

        if conditions_met:
            print(f"\n在第 {attempt} 次尝试中找到满足所有条件的解！")
            print("\n找到的解：")
            print(f"p = {p:.6f} [固定值]")
            print(f"q = {q:.6f} [固定值]")
            print(f"beta1 = {beta1:.6f} 弧度 ({math.degrees(beta1):.2f} 度) [固定值]")
            print(f"beta2 = {beta2:.6f} 弧度 ({math.degrees(beta2):.2f} 度) [固定值]")
            print(f"mu1 = {mu1:.6f} 弧度 ({math.degrees(mu1):.2f} 度)")
            print(f"mu2 = {mu2:.6f} 弧度 ({math.degrees(mu2):.2f} 度)")
            print(f"alpha = {alpha:.6f}")
            print(f"theta = {theta:.6f} 弧度 ({math.degrees(theta):.2f} 度)")

            print("\n验证结果：")
            print(f"lambda2 = {lambda2:.6f}")
            print(f"lambda3 = {lambda3:.6f}")
            print(f"lambda2 - lambda3 = {lambda2 - lambda3:.6e}")
            print(f"equation_1 = {eq1:.6e}")
            print(f"equation_2 = {eq2:.6e}")
            print(f"lambda平均大小 = {lambda_magnitude:.6f}")

            exit()  # 找到满足条件的解，退出程序

    # 每100次尝试保存一次最佳结果
    if attempt % 100 == 0:
        print(f"\n第 {attempt} 次尝试后仍未找到完美解，当前最佳结果：")
        if best_result is not None:
            p, q, beta1, beta2, mu1, mu2, alpha, theta, lambda2, lambda3, eq1, eq2 = best_result
            print(f"lambda2 - lambda3 = {lambda2 - lambda3:.6e}")
            print(f"equation_1 = {eq1:.6e}")
            print(f"equation_2 = {eq2:.6e}")
        else:
            print("尚未找到任何可行解")
        print("继续尝试...\n")

# 如果达到最大尝试次数仍未找到完美解，输出最佳结果
print(f"\n达到最大尝试次数 ({max_attempts})，未找到完全满足条件的解")
if best_result is not None:
    print("\n最佳结果：")
    p, q, beta1, beta2, mu1, mu2, alpha, theta, lambda2, lambda3, eq1, eq2 = best_result
    print(f"p = {p:.6f} [固定值]")
    print(f"q = {q:.6f} [固定值]")
    print(f"beta1 = {beta1:.6f} 弧度 ({math.degrees(beta1):.2f} 度) [固定值]")
    print(f"beta2 = {beta2:.6f} 弧度 ({math.degrees(beta2):.2f} 度) [固定值]")
    print(f"mu1 = {mu1:.6f} 弧度 ({math.degrees(mu1):.2f} 度)")
    print(f"mu2 = {mu2:.6f} 弧度 ({math.degrees(mu2):.2f} 度)")
    print(f"alpha = {alpha:.6f}")
    print(f"theta = {theta:.6f} 弧度 ({math.degrees(theta):.2f} 度)")

    print("\n验证结果：")
    print(f"lambda2 = {lambda2:.6f}")
    print(f"lambda3 = {lambda3:.6f}")
    print(f"lambda2 - lambda3 = {lambda2 - lambda3:.6e}")
    print(f"equation_1 = {eq1:.6e}")
    print(f"equation_2 = {eq2:.6e}")
    lambda_magnitude = (lambda2 ** 2 + lambda3 ** 2) / 2
    print(f"lambda平均大小 = {lambda_magnitude:.6f}")
else:
    print("未找到任何可行解")
