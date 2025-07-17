from sympy import symbols, sin, asin, diff, Eq, pi, simplify, trigsimp, pprint

# 定义符号变量
X = symbols('A0 A1 B0 B1 E00 E10 E11')  # 点坐标参数
x = symbols('x0 x1 x2 x3 x4 x5 x6 x7')    # 切平面变量

# 定义 theta 和 g
theta = (
    asin((X[4] + X[2]) / (1 + X[0])) +
    asin((X[5] + X[2]) / (1 + X[1])) +
    asin((X[6] + X[3]) / (1 + X[1])) - pi
)
g = sin(theta) * (1 + X[0]) - X[3]

# 计算法向量（符号形式）
normal = [diff(g, X[i]) for i in range(7)] + [-1]

# 构造切平面方程
tangent_plane_eq = Eq(
    sum(normal[i] * (x[i] - X[i]) for i in range(7)) + normal[7] * (x[7] - g),
    0
)

print("符号形式的切平面方程:")
simplified_eq = trigsimp(tangent_plane_eq)
pprint(simplified_eq)