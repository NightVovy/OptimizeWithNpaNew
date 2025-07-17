import numpy as np
from math import cos, sin, pi, sqrt


def calculate_expectations(theta, mu, beta):
    # 直接使用解析表达式计算期望值
    EA0 = cos(2 * theta)
    EA1 = 0.0
    EB0 = cos(2 * theta) * cos(mu)
    EB1 = -cos(2 * theta) * cos(mu)

    EA0B0 = cos(mu)
    EA0B1 = -cos(mu)
    EA1B0 = sin(2 * theta) * sin(mu)
    EA1B1 = sin(2 * theta) * sin(mu)

    # 计算i_beta不等式相关量
    i_beta = beta * (EA0B0 - EA0B1) + EA1B0 + EA1B1
    bound = 2 * sqrt(beta ** 2 + sin(2 * theta) ** 2)

    # 输出结果
    print("单算符期望值:")
    print(f"<A0> = {EA0:.4f}")
    print(f"<A1> = {EA1:.4f}")
    print(f"<B0> = {EB0:.4f}")
    print(f"<B1> = {EB1:.4f}")

    print("\n联合期望值:")
    print(f"<A0B0> = {EA0B0:.4f}")
    print(f"<A0B1> = {EA0B1:.4f}")
    print(f"<A1B0> = {EA1B0:.4f}")
    print(f"<A1B1> = {EA1B1:.4f}")

    print("\ni_beta不等式测试:")
    print(f"beta*(<A0B0>-<A0B1>)+<A1B0>+<A1B1> = {i_beta:.4f}")
    print(f"2*sqrt(beta^2 + sin(2θ)^2) = {bound:.4f}")
    print(f"是否违背i_beta不等式: {'是' if i_beta > bound else '否'}")

    return EA0, EA1, EB0, EB1, EA0B0, EA0B1, EA1B0, EA1B1

theta = pi / 4
mu = pi / 4
beta = 0.6


# 调用函数获取期望值
EA0, EA1, EB0, EB1, EA0B0, EA0B1, EA1B0, EA1B1 = calculate_expectations(
    theta=theta,
    mu=mu,
    beta=beta
)