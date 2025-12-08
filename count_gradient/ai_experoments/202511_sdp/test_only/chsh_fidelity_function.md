记录该公式的写法。

投影算子强调：
# 根据 m_vec=[2, 2] 确定全局索引
# Alice
SETTING_A0 = 0
SETTING_A1 = 1

# Bob (偏移量 = Alice的总设置数 = 2)
SETTING_B0 = 2
SETTING_B1 = 3

# Alice 的序列: E0 -> E1 -> E0
# 对应列表: [0, 1, 0]
seq_alice_complex = [SETTING_A0, SETTING_A1, SETTING_A0]

# Bob 的序列: F0 -> F1 -> F0
# 对应列表: [2, 3, 2]
seq_bob_complex = [SETTING_B0, SETTING_B1, SETTING_B0]


# 获取项 < E0 E1 E0 * F0 F1 F0 >
term_E0E1E0_F0F1F0 = get_gamma_element(
    G,              # CVXPY 变量
    ind_catalog,    # 索引目录
    m_vec,          # [2, 2]
    seq_alice_complex, # 输入 [0, 1, 0]
    seq_bob_complex    # 输入 [2, 3, 2]
)

# 检查是否找到 (如果层级 K 不够高，比如 K=2，长度为3的项可能找不到)
if term_E0E1E0_F0F1F0 is None:
    raise ValueError("K 值太小，无法在 Gamma 矩阵中找到 E0E1E0*F0F1F0 这一项。")

-------------------------------------------------


func = 1/2 + <CHSH> * 1/(2*np.sqrt(2))
    - 1/8 * (d_A0A1B0B1 - d_A0A1B1B0 - d_A1A0B0B1 + d_A1A0B1B0)
    + 1/(8*np.sqrt(2)) * (3 * d_A1B1 - 2 * d_A0B1 - 2 * d_A1B0
        + d_A0A1A0B1 - 2 * d_A0A1A0B0 + d_A1B0B1B0 - 2 * d_A0B0B1B0
        - d_A0A1A0B0B1B0)     

d_A0B0 = 4 * term_E0_F0 - 2 * term_E0 - 2 * term_F0 + term_I
d_A1B1 = 4 * term_E1F1 - 2 * term_E1 - 2 * term_F1 + term_I
d_A0B1 = 4 * term_E0F1 - 2 * term_E0 - 2 * term_F1 + term_I
d_A1B0 = 4 * term_E1F0 - 2 * term_E1 - 2 * term_F0 + term_I

d_A0A1A0B1 = 16 * term_E0E1E0F1 - 8 * (term_E0E1F1 + term_E1E0F1 + term_E0E1E0) + 4 * (term_E0E1 + term_E1E0 + term_E1F1)
            - 2 * (term_E1 + term_F1) + term_I

d_A0A1A0B0 = 16 * term_E0E1E0F0 - 8 * (term_E0E1F0 + term_E1E0F0 + term_E0E1E0) + 4 * (term_E0E1 + term_E1E0 + term_E1F0)
            - 2 * (term_E1 + term_F0) + term_I

d_A1B0B1B0 = 16 * term_E1F0F1F0 - 8 * (term_F0F1F0 + term_E1F0F1 + term_E1F1F0) + 4 * (term_E1F1 + term_F0F1 + term_F1F0)
            - 2 * (term_E1 + term_F1) + term_I

d_A0B0B1B0 = 16 * term_E0F0F1F0 - 8 * (term_F0F1F0 + term_E0F0F1 + term_E0F1F0) + 4 * (term_E0F1 + term_F0F1 + term_F1F0)
            - 2 * (term_E0 + term_F1) + term_I

d_A0A1B0B1 = 16 * term_E0E1F0F1 - 8 * (term_E0E1F0 + term_E0E1F1 + term_E0F0F1 + term_E1F0F1)
            + 4 * (term_E0F0 + term_E0F1 + term_E1F0 + term_E1F1)
            + 4 * (term_E0E1 + term_F0F1)
            - 2 * (term_E0 + term_E1 + term_F0 + term_F1)
            + term_I

d_A0A1B1B0 = 16 * term_E0E1F1F0 - 8 * (term_E0E1F1 + term_E0E1F0 + term_E0F1F0 + term_E1F1F0)
            + 4 * (term_E0F1 + term_E0F0 + term_E1F1 + term_E1F0)
            + 4 * (term_E0E1 + term_F1F0)
            - 2 * (term_E0 + term_E1 + term_F1 + term_F0)
            + term_I

d_A1A0B0B1 = 16 * term_E1E0F0F1 - 8 * (term_E1E0F0 + term_E1E0F1 + term_E1F0F1 + term_E0F0F1)
            + 4 * (term_E1F0 + term_E1F1 + term_E0F0 + term_E0F1)
            + 4 * (term_E1E0 + term_F0F1)
            - 2 * (term_E1 + term_E0 + term_F0 + term_F1)
            + term_I 

d_A1A0B1B0 = 16 * term_E1E0F1F0 - 8 * (term_E1E0F1 + term_E1E0F0 + term_E1F1F0 + term_E0F1F0)
            + 4 * (term_E1F1 + term_E1F0 + term_E0F1 + term_E0F0)
            + 4 * (term_E1E0 + term_F1F0)
            - 2 * (term_E1 + term_E0 + term_F1 + term_F0)
            + term_I

d_A0A1A0B0B1B0 = 64 * term_E0E1E0F0F1F0
            - 32 * (term_E0E1E0F0F1 + term_E0E1E0F1F0 + term_E0E1F0F1F0 + term_E1E0F0F1F0)
            + 16 * (term_E0E1E0F1 + term_E1F0F1F0 + term_E0E1F0F1 + term_E0E1F1F0 + term_E1E0F0F1 + term_E1E0F1F0)
            - 8 * (term_E0E1E0 + term_F0F1F0 + term_E0E1F1 + term_E1E0F1 + term_E1F0F1 + term_E1F1F0)
            + 4 * (term_E0E1 + term_E1E0 + term_F0F1 + term_F1F0 + term_E1F1)
            - 2 * (term_E1 + term_F1)
            + term_I






