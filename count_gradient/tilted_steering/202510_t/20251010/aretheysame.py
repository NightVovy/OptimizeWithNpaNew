import math


def calculate_lambda(p00, p01, p10, p11, beta1, beta2, mu1, mu2, theta, alpha):
    """
    计算λ值的函数

    参数:
    p00, p01, p10, p11: p系数
    beta1, beta2: β角度（弧度）
    mu1, mu2: μ角度（弧度）
    theta: θ角度（弧度）
    alpha: α系数

    返回:
    lambda_value: 计算结果
    """

    # 第一部分：(p00*cosβ1 + p10*cosβ2)*cosμ1 + (p01*cosβ1 - p11*cosβ2)*cosμ2
    part1 = ((p00 * math.cos(beta1) + p10 * math.cos(beta2)) * math.cos(mu1) +
             (p01 * math.cos(beta1) - p11 * math.cos(beta2)) * math.cos(mu2))

    # 第二部分：[(p00*sinβ1 + p10*sinβ2)*sinμ1 + (p01*sinβ1 - p11*sinβ2)*sinμ2] * sin2θ
    part2 = ((p00 * math.sin(beta1) + p10 * math.sin(beta2)) * math.sin(mu1) +
             (p01 * math.sin(beta1) - p11 * math.sin(beta2)) * math.sin(mu2)) * math.sin(2 * theta)

    # 第三部分：α * cos2θ * cosβ1
    part3 = alpha * math.cos(2 * theta) * math.cos(beta1)

    # 总和
    lambda_value = part1 + part2 + part3

    return lambda_value


def calculate_lambda_2(p00, p01, p10, p11, beta1, beta2, mu1, mu2, theta, alpha):
    """
    计算λ值的第二个函数

    参数:
    p00, p10, p11, p101: p系数
    beta1, beta2: β角度（弧度）
    mu1, mu2: μ角度（弧度）
    theta: θ角度（弧度）
    alpha: α系数

    返回:
    lambda_value_2: 计算结果
    """

    # 第一部分：α * cosβ1
    part1 = alpha * math.cos(beta1)

    # 第二部分：(p10*sinμ1 - p11*sinμ2) * sinβ2 * (sinθ/cosθ)
    part2 = (p10 * math.sin(mu1) - p11 * math.sin(mu2)) * math.sin(beta2) * (math.sin(theta) / math.cos(theta))

    # 第三部分：(p00*sinμ1 + p101*sinμ2) * sinβ1 * (sinθ/cosθ)
    part3 = (p00 * math.sin(mu1) + p01 * math.sin(mu2)) * math.sin(beta1) * (math.sin(theta) / math.cos(theta))

    # 总和
    lambda_value_2 = part1 + part2 + part3

    return lambda_value_2

def calculate_lambda_3(p00, p01, p10, p11, beta1, beta2, mu1, mu2, theta, alpha):
    """
    计算λ值的第二个函数

    参数:
    p00, p10, p11, p101: p系数
    beta1, beta2: β角度（弧度）
    mu1, mu2: μ角度（弧度）
    theta: θ角度（弧度）
    alpha: α系数

    返回:
    lambda_value_2: 计算结果
    """

    # 第一部分：α * cosβ1
    part1 = - alpha * math.cos(beta1)

    # 第二部分：(p10*sinμ1 - p11*sinμ2) * sinβ2 * (sinθ/cosθ)
    part2 = (p10 * math.sin(mu1) - p11 * math.sin(mu2)) * math.sin(beta2) * (math.cos(theta) / math.sin(theta))

    # 第三部分：(p00*sinμ1 + p101*sinμ2) * sinβ1 * (sinθ/cosθ)
    part3 = (p00 * math.sin(mu1) + p01 * math.sin(mu2)) * math.sin(beta1) * (math.cos(theta) / math.sin(theta))

    # 总和
    lambda_value_3 = part1 + part2 + part3

    return lambda_value_3

# 等于0
def equation_1(p00, p01, p10, p11, beta1, beta2, mu1, mu2, alpha):
    """
    计算第一个方程的值

    参数:
    p00, p01, p10, p11: p系数
    beta1, beta2: β角度（弧度）
    mu1, mu2: μ角度（弧度）
    alpha: α系数

    返回:
    eq1_value: 第一个方程的计算结果（应该等于0）
    """

    # α * sinβ₁
    part1 = alpha * math.sin(beta1)

    # (p₀₀cosβ₁ + p₁₀cosβ₂) * sinμ₁
    part2 = (p00 * math.cos(beta1) + p10 * math.cos(beta2)) * math.sin(mu1)

    # (p₀₁cosβ₁ - p₁₁cosβ₂) * sinμ₂
    part3 = (p01 * math.cos(beta1) - p11 * math.cos(beta2)) * math.sin(mu2)

    # 方程总和（应该等于0）
    eq1_value = part1 + part2 + part3

    return eq1_value


def equation_2(p00, p01, p10, p11, beta1, beta2, mu1, mu2):
    """
    计算第二个方程的值

    参数:
    p00, p01, p10, p11: p系数
    beta1, beta2: β角度（弧度）
    mu1, mu2: μ角度（弧度）

    返回:
    eq2_value: 第二个方程的计算结果（应该等于0）
    """

    # (p₀₀sinβ₁ + p₁₀sinβ₂) * cosμ₁
    part1 = (p00 * math.sin(beta1) + p10 * math.sin(beta2)) * math.cos(mu1)

    # (p₀₁sinβ₁ - p₁₁sinβ₂) * cosμ₂
    part2 = (p01 * math.sin(beta1) - p11 * math.sin(beta2)) * math.cos(mu2)

    # 方程总和（应该等于0）
    eq2_value = part1 + part2

    return eq2_value



# 使用示例（假设所有角度都需要转换为弧度）
# 如果输入是角度，需要先转换为弧度
p = 0.6
q = 0.7
p00 = p * q
p01 = p * (1-q)
p10 = (1-p) * q
p11 = (1-p) * (1-q)

beta1_deg, beta2_deg, mu1_deg, mu2_deg, theta_deg = 0, 30, 40, 75, 25  # 示例角度值
alpha = 0.6  # 示例值

# 转换为弧度
beta1 = math.radians(beta1_deg)
beta2 = math.radians(beta2_deg)
mu1 = math.radians(mu1_deg)
mu2 = math.radians(mu2_deg)
theta = math.radians(theta_deg)

result1 = calculate_lambda(p00, p01, p10, p11, beta1, beta2, mu1, mu2, theta, alpha)
result2 = calculate_lambda_2(p00, p01, p10, p11, beta1, beta2, mu1, mu2, theta, alpha)
result3 = calculate_lambda_3(p00, p01, p10, p11, beta1, beta2, mu1, mu2, theta, alpha)
eq1 = equation_1(p00, p01, p10, p11, beta1, beta2, mu1, mu2, alpha)
eq2 = equation_2(p00, p01, p10, p11, beta1, beta2, mu1, mu2)

print(f"λ = {result1}")
print(f"λ2 = {result2}")
print(f"λ3 = {result3}")
print(f"方程1的值: {eq1}")
print(f"方程2的值: {eq2}")