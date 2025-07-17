import math

# 定义常量
A0 = 1/2
A1 = 0
B0 = 1 / math.sqrt(7)
B1 = 1 / math.sqrt(7)
E00 = 2 / math.sqrt(7)
E01 = 2 / math.sqrt(7)
E10 = 3 / (2 * math.sqrt(7))
E11 = -3 / (2 * math.sqrt(7))
s = - 1
t = - 1

# 计算各项
term1 = math.asin((E00 + s * B0) / (1 + s * A0))
term2 = math.asin((E01 + s * B1) / (1 + s * A0))
term3 = math.asin((E10 + t * B0) / (1 + t * A1))
term4 = math.asin((E11 + t * B1) / (1 + t * A1))

# 计算总和
result = term1 + term2 + term3 - term4

# 输出结果（以弧度表示）
print("计算结果（弧度）:", result)

