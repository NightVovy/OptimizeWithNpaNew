记录该公式的写法。
alpha * A0 + A0B0 + A0B1 + A1B0 - A1B1

最大量子违背：sqrt(8 + 2 * alpha^2)

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

【备注: 下面的我都没验证过 如果出问题就来算一下】
d_A0 = 2 * term_E0 - term_I
d_B0 = 2 * term_F0 - term_I

d_A0B0 = 4 * term_E0_F0 - 2 * term_E0 - 2 * term_F0 + term_I
d_A1B2 = 4 * term_E1_F2 - 2 * term_E1 - 2 * term_F2 + term_I

d_A1B0B2B0 = 16 * term_E1_F0F2F0 - 8 * (term_E1_F0F2 + term_E1_F2F0 + term_F0F2F0) 
        + 4 * (term_E1_F2 + term_F0F2 + term_F2F0) 
        - 2 * (term_E1 + term_F2) + term_I

d_A0A1A0B2 = 16 * term_E0E1E0_F2 - 8 * (term_E0E1_F2 + term_E1E0_F2 + term_E0E1E0) 
        + 4 * (term_E1_F2 + term_E0E1 + term_E1E0)
        - 2 * (term_E1 + term_F2) + term_I

d_A0A1B0B2 = 16 * term_E0E1_F0F2 
        - 8 * (term_E0E1_F0 + term_E0E1_F2 + term_E0_F0F2 + term_E1_F0F2) 
        + 4 * (term_E0E1 + term_F0F2 + term_E0_F0 + term_E0_F2 + term_E1_F0 + term_E1_F2)
        - 2 * (term_E0 + term_E1 + term_F0 + term_F2) + term_I

d_A0A1B2B0 = 16 * term_E0E1_F2F0 
        - 8 * (term_E0E1_F2 + term_E0E1_F0 + term_E0_F2F0 + term_E1_F2F0) 
        + 4 * (term_E0E1 + term_F2F0 + term_E0_F2 + term_E0_F0 + term_E1_F2 + term_E1_F0)
        - 2 * (term_E0 + term_E1 + term_F2 + term_F0) + term_I

d_A1A0B0B2 = 16 * term_E1E0_F0F2 
        - 8 * (term_E1E0_F0 + term_E1E0_F2 + term_E1_F0F2 + term_E0_F0F2) 
        + 4 * (term_E1E0 + term_F0F2 + term_E1_F0 + term_E1_F2 + term_E0_F0 + term_E0_F2)
        - 2 * (term_E1 + term_E0 + term_F0 + term_F2) + term_I

d_A1A0B2B0 = 16 * term_E1E0_F2F0 
        - 8 * (term_E1E0_F2 + term_E1E0_F0 + term_E1_F2F0 + term_E0_F2F0) 
        + 4 * (term_E1E0 + term_F2F0 + term_E1_F2 + term_E1_F0 + term_E0_F2 + term_E0_F0)
        - 2 * (term_E1 + term_E0 + term_F2 + term_F0) + term_I

d_A0A1A0B0B2B0 = 64 * (term_E0E1E0_F0F2F0) 
        - 32 * (term_E0E1E0_F0F2 + term_E0E1E0_F2F0 + term_E0E1_F0F2F0 + term_E1E0_F0F2F0)
        + 16 * (term_E0E1E0_F2 + term_E1_F0F2F0 + term_E0E1_F0F2 + term_E0E1_F2F0 + term_E1E0_F0F2 + term_E1E0_F2F0)
        - 8 * (term_E0E1E0 + term_F0F2F0 + term_E0E1_F2 + term_E1E0_F2 + term_E1_F0F2 + term_E1_F2F0)
        + 4 * (term_E0E1 + term_E1E0 + term_F0F2 + term_F2F0 + term_E1_F2)
        - 2 * (term_E1 + term_F2) + term_I


--- 3x3
term_E0E1E0_F0F2F0

--- 3x2
term_E0E1E0_F0F2

term_E0E1E0_F2F0

term_E0E1_F0F2F0

term_E1E0_F0F2F0

--- 1+3 / 2+2 / 3+1
term_E0E1E0_F2

term_E1_F0F2F0

term_E0E1_F0F2

term_E0E1_F2F0

term_E1E0_F0F2

term_E1E0_F2F0

--- 2+1 / 1+2
term_E0E1_F0

term_E0E1_F2

term_E1E0_F0

term_E1E0_F2

term_E0_F0F2

term_E0_F2F0

term_E1_F0F2

term_E1_F2F0

--- 3
term_E0E1E0(右侧为I)

term_F0F2F0(左侧为I)

--- 2 / 1+1
term_E1_F2

term_E0_F0

term_E0_F2

term_E1_F0

term_E0E1, term_E1E0, (右侧为I)

term_F0F2, term_F2F0 (左侧为I)

--- 1
term_E0 (右侧为I)

term_E1 (右侧为I)

term_F0 (左侧为I)

term_F2 (左侧为I)

term_I


