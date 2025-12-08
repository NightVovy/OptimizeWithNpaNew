d_I = term_I
d_B0 = 2 * term_F0 - term_I
d_B1 = 2 * term_F1 - term_I
d_B2 = 2 * term_F2 - term_I

# --- 2-Body ---
d_B0B1 = 4 * term_F0F1 - 2 * term_F0 - 2 * term_F1 + term_I
d_B1B0 = 4 * term_F1F0 - 2 * term_F1 - 2 * term_F0 + term_I
d_B0B2 = 4 * term_F0F2 - 2 * term_F0 - 2 * term_F2 + term_I
d_B2B0 = 4 * term_F2F0 - 2 * term_F2 - 2 * term_F0 + term_I
d_B1B2 = 4 * term_F1F2 - 2 * term_F1 - 2 * term_F2 + term_I
d_B2B1 = 4 * term_F2F1 - 2 * term_F2 - 2 * term_F1 + term_I

# --- 3-Body ---
# d_B2B1B0
d_B2B1B0 = (8 * term_F2_F1F0
                - 4 * (term_F2F1 + term_F2F0 + term_F1F0)
                + 2 * (term_F2 + term_F1 + term_F0) - term_I)
# d_B2B0B1
d_B2B0B1 = (8 * term_F2_F0F1
                - 4 * (term_F2F0 + term_F2F1 + term_F0F1)
                + 2 * (term_F2 + term_F0 + term_F1) - term_I)

# d_B2B0B2 (简化为 <F2 F0 F2> 相关的项) ************* 已简化
d_B2B0B2 = 8 * term_F2_F0F2
                - 4 * (term_F2F0 + term_F0F2)
                + 2 * term_F0 - term_I

# d_B2B1B2
d_B2B1B2 = 8 * term_F2_F1F2
                - 4 * (term_F2F1  + term_F1F2)
                + 2 * term_F1  - term_I
# d_B0B2B0
d_B0B2B0 = 8 * term_F0_F2F0
                - 4 * (term_F0F2  + term_F2F0)
                + 2 * term_F2  - term_I
# d_B0B2B1
d_B0B2B1 = (8 * term_F0_F2F1
                - 4 * (term_F0F2 + term_F0F1 + term_F2F1)
                + 2 * (term_F0 + term_F2 + term_F1) - term_I)
# d_B1B2B0
d_B1B2B0 = (8 * term_F1_F2F0
                - 4 * (term_F1F2 + term_F1F0 + term_F2F0)
                + 2 * (term_F1 + term_F2 + term_F0) - term_I)
# d_B1B2B1
d_B1B2B1 = 8 * term_F1_F2F1
                - 4 * (term_F1F2  + term_F2F1)
                + 2 * term_F2  - term_I

    # --- 4-Body ---
    # d_B0B2B1B0
    # Formula: 16*F0F2F1F0 - 8*(F0F2F1 + F0F2F0 + F0F1F0 + F2F1F0)
    #          + 4*(F0F2 + F0F1 + F2F1 + F2F0 + F1F0) - 2*(F2 + F1) + I
d_B0B2B1B0 = (16 * term_F0F2_F1F0
                  - 8 * (term_F0_F2F1 + term_F0_F2F0 + term_F0_F1F0 + term_F2_F1F0)
                  + 4 * (term_F0F2 + term_F0F1 + term_F2F1 + term_F2F0 + term_F1F0)
                  - 2 * (term_F2 + term_F1)
                  + term_I)
    # d_B0B2B0B1
    # Formula: 16*F0F2F0F1 - 8*(F0F2F0 + F0F2F1 + F2F0F1)
    #          + 4*(F0F2 + F2F0 + F2F1) - 2*(F2 + F1) + I
d_B0B2B0B1 = (16 * term_F0F2_F0F1
                  - 8 * (term_F0_F2F0 + term_F0_F2F1 + term_F2_F0F1)
                  + 4 * (term_F0F2 + term_F2F0 + term_F2F1)
                  - 2 * (term_F2 + term_F1)
                  + term_I)
    # d_B0B2B0B2
    # Formula: 16*F0F2F0F2 - 8*(F0F2F0 + F2F0F2)
    #          + 4*(F0F2 + F2F0) + I
d_B0B2B0B2 = (16 * term_F0F2_F0F2
                  - 8 * (term_F0_F2F0 + term_F2_F0F2)
                  + 4 * (term_F0F2 + term_F2F0)
                  + term_I)

    # d_B0B2B1B2 * **********************已检查 已修正
    # Formula: 16*F0F2F1F2 - 8*(F0F2F1 + F0F1F2 + F2F1F2)
    #          + 4*(-F0F2 + F0F1 + F2F1 + F1F2) - 2*(F0 + F1) + I
    # 注意: F0F2 前是负号
d_B0B2B1B2 = (16 * term_F0F2_F1F2
    - 8 * (term_F0_F2F1 + term_F0_F1F2 + term_F2_F1F2)
    + 4 * (term_F0F1 + term_F2F1 + term_F1F2)
    - 2 * (term_F0 + term_F1)
    + term_I)


    # d_B1B2B1B0
    # Formula: 16*F1F2F1F0 - 8*(F1F2F1 + F1F2F0 + F2F1F0)
    #          + 4*(F1F2 + F2F1 + F2F0) - 2*(F2 + F0) + I
d_B1B2B1B0 = (16 * term_F1F2_F1F0
                  - 8 * (term_F1_F2F1 + term_F1_F2F0 + term_F2_F1F0)
                  + 4 * (term_F1F2 + term_F2F1 + term_F2F0)
                  - 2 * (term_F2 + term_F0)
                  + term_I)

    # d_B1B2B0B1
    # Formula: 16*F1F2F0F1 - 8*(F1F2F0 + F1F2F1 + F2F0F1)
    #          + 4*(F1F2 + F2F0 + F2F1) - 2*(F2 + F0) + I
d_B1B2B0B1 = (16 * term_F1F2_F0F1
                  - 8 * (term_F1_F2F0 + term_F1_F2F1 + term_F2_F0F1)
                  + 4 * (term_F1F2 + term_F2F0 + term_F2F1)
                  - 2 * (term_F2 + term_F0)
                  + term_I)

    # d_B1B2B0B2 * **************已检查 已修正
    # Formula: 16*F1F2F0F2 - 8*(F1F2F0 + F1F0F2 + F2F0F2)
    #          + 4*(-F1F2 + F1F0 + F2F0 + F0F2) - 2*(F1 + F0) + I
    # 注意: F1F2 前是负号
d_B1B2B0B2 = (16 * term_F1F2_F0F2
                  - 8 * (term_F1_F2F0 + term_F1_F0F2 + term_F2_F0F2)
                  + 4 * (term_F1F0 + term_F2F0 + term_F0F2)
                  - 2 * (term_F1 + term_F0)
                  + term_I)
    # d_B1B2B1B2
    # Formula: 16*F1F2F1F2 - 8*(F1F2F1 + F2F1F2)
    #          + 4*(F1F2 + F2F1) + I
d_B1B2B1B2 = (16 * term_F1F2_F1F2
                  - 8 * (term_F1_F2F1 + term_F2_F1F2)
                  + 4 * (term_F1F2 + term_F2F1)
                  + term_I)

--- 3x3


--- 3x2

--- 4/ 1+3 / 2+2 / 3+1
term_F0F2_F1F0
term_F0F2_F0F1
term_F0F2_F0F2 
term_F0F2_F1F2 

term_F1F2_F1F0 
term_F1F2_F0F1 
term_F1F2_F0F2 
term_F1F2_F1F2 



--- 3 /  2+1 / 1+2
term_F0_F2F0 
term_F0_F2F1
term_F1_F2F0
term_F1_F2F1
term_F1_F0F2

term_F2_F0F1 
term_F2_F0F2 
term_F2_F1F0
term_F2_F1F2

term_F0_F1F0 
term_F0_F1F2 

--- 2 / 1+1
term_F0F1 
term_F1F0 
term_F0F2 
term_F2F0 
term_F1F2 
term_F2F1 

--- 1
term_I 
term_F0 
term_F1 
term_F2 