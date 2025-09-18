import sympy as sp

# 定义符号变量
theta, a0, a1, b0, b1 = sp.symbols('theta a0 a1 b0 b1')

# 定义相关函数
def corre_A0(theta_val, a0_val):
    return sp.cos(2*theta_val) * sp.cos(a0_val)

def corre_A1(theta_val, a1_val):
    return sp.cos(2*theta_val) * sp.cos(a1_val)

def corre_B0(theta_val, b0_val):
    return sp.cos(2*theta_val) * sp.cos(b0_val)

def corre_B1(theta_val, b1_val):
    return sp.cos(2*theta_val) * sp.cos(b1_val)

def corre_A0B0(theta_val, a0_val, b0_val):
    return sp.cos(a0_val)*sp.cos(b0_val) + sp.sin(2*theta_val)*sp.sin(a0_val)*sp.sin(b0_val)

def corre_A0B1(theta_val, a0_val, b1_val):
    return sp.cos(a0_val)*sp.cos(b1_val) + sp.sin(2*theta_val)*sp.sin(a0_val)*sp.sin(b1_val)

def corre_A1B0(theta_val, a1_val, b0_val):
    return sp.cos(a1_val)*sp.cos(b0_val) + sp.sin(2*theta_val)*sp.sin(a1_val)*sp.sin(b0_val)

def corre_A1B1(theta_val, a1_val, b1_val):
    return sp.cos(a1_val)*sp.cos(b1_val) + sp.sin(2*theta_val)*sp.sin(a1_val)*sp.sin(b1_val)

# 定义向量P
P = sp.Matrix([
    corre_A0(theta, a0),
    corre_A1(theta, a1),
    corre_B0(theta, b0),
    corre_B1(theta, b1),
    corre_A0B0(theta, a0, b0),
    corre_A0B1(theta, a0, b1),
    corre_A1B0(theta, a1, b0),
    corre_A1B1(theta, a1, b1)
])

print("="*60)
print("向量 P 的表达式:")
print("="*60)
for i, comp in enumerate(P):
    print(f"P[{i}] = {comp}")
print()

# 计算各个偏导数
print("="*60)
print("各个偏导数:")
print("="*60)

# 关于theta的偏导
dP_dtheta = sp.Matrix([sp.diff(comp, theta) for comp in P])
print("\n∂P/∂theta:")
for i, comp in enumerate(dP_dtheta):
    simplified = sp.simplify(comp)
    print(f"∂P[{i}]/∂theta = {simplified}")

# 关于a0的偏导
dP_da0 = sp.Matrix([sp.diff(comp, a0) for comp in P])
print("\n∂P/∂a0:")
for i, comp in enumerate(dP_da0):
    simplified = sp.simplify(comp)
    print(f"∂P[{i}]/∂a0 = {simplified}")

# 关于a1的偏导
dP_da1 = sp.Matrix([sp.diff(comp, a1) for comp in P])
print("\n∂P/∂a1:")
for i, comp in enumerate(dP_da1):
    simplified = sp.simplify(comp)
    print(f"∂P[{i}]/∂a1 = {simplified}")

# 关于b0的偏导
dP_db0 = sp.Matrix([sp.diff(comp, b0) for comp in P])
print("\n∂P/∂b0:")
for i, comp in enumerate(dP_db0):
    simplified = sp.simplify(comp)
    print(f"∂P[{i}]/∂b0 = {simplified}")

# 关于b1的偏导
dP_db1 = sp.Matrix([sp.diff(comp, b1) for comp in P])
print("\n∂P/∂b1:")
for i, comp in enumerate(dP_db1):
    simplified = sp.simplify(comp)
    print(f"∂P[{i}]/∂b1 = {simplified}")

# 构造完整的Jacobian矩阵
print("="*60)
print("完整的 Jacobian 矩阵 (8×5):")
print("="*60)

variables = [theta, a0, a1, b0, b1]
jacobian_matrix = sp.Matrix.zeros(8, 5)

for i, var in enumerate(variables):
    for j in range(8):
        derivative = sp.diff(P[j], var)
        jacobian_matrix[j, i] = sp.simplify(derivative)

# 显示Jacobian矩阵
print("Jacobian矩阵结构:")
print("行: P[0]到P[7] (8个分量)")
print("列: ∂/∂theta, ∂/∂a0, ∂/∂a1, ∂/∂b0, ∂/∂b1")
print()

for i in range(8):
    for j in range(5):
        expr = jacobian_matrix[i, j]
        print(f"J[{i},{j}] = ∂P[{i}]/∂{variables[j]} = {expr}")
    print()

print("="*60)
print("Jacobian矩阵的简化版本:")
print("="*60)

# 进一步简化整个矩阵
simplified_jacobian = sp.Matrix([[sp.simplify(jacobian_matrix[i, j])
                                for j in range(5)] for i in range(8)])

for i in range(8):
    for j in range(5):
        expr = simplified_jacobian[i, j]
        print(f"J_simplified[{i},{j}] = {expr}")
    print()

# 验证偏导数的正确性（符号验证）
print("="*60)
print("偏导数验证:")
print("="*60)

# 检查一些基本的偏导数关系
print("验证几个典型的偏导数:")
print(f"∂(cos(2θ)cos(a0))/∂theta = {sp.diff(sp.cos(2*theta)*sp.cos(a0), theta)}")
print(f"∂(cos(2θ)cos(a0))/∂a0 = {sp.diff(sp.cos(2*theta)*sp.cos(a0), a0)}")
print(f"∂(sin(2θ)sin(a0)sin(b0))/∂theta = {sp.diff(sp.sin(2*theta)*sp.sin(a0)*sp.sin(b0), theta)}")