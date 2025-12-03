记录该公式的写法。

投影算子强调：
# 根据 m_vec=[2, 3] 确定全局索引
# Alice
SETTING_A0 = 0
SETTING_A1 = 1

# Bob (偏移量 = Alice的总设置数 = 2)
SETTING_B0 = 2
SETTING_B1 = 3
SETTING_B2 = 4

输入: alpha = 1/4
alpha 和 mu, theta的关系:
sin(2theta) = sqrt((1 - alpha^2/4) / (1 + alpha^2/4))
tan(mu) = sin(2theta)


func = 1/2 * (1 + d_A0B0 + cos2theta * (d_A0 + d_B0))
    + sin2theta/8 * (d_A1B2 - d_A1B0B2B0 - d_A0A1A0B2 + d_A0A1A0B0B2B0)
    - sin2theta/8 * (d_A0A1B0B2 - d_A0A1B2B0 - d_A1A0B0B2 + d_A1A0B2B0)