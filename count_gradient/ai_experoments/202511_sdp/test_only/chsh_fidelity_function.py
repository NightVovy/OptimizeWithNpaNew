# 2-body terms
d_A1B1 = 4 * term_E1_F1 - 2 * term_E1 - 2 * term_F1 + term_I
d_A0B1 = 4 * term_E0_F1 - 2 * term_E0 - 2 * term_F1 + term_I
d_A1B0 = 4 * term_E1_F0 - 2 * term_E1 - 2 * term_F0 + term_I

# 4-body terms (Mixed 3-1 split)
d_A0A1A0B1 = 16 * term_E0E1E0_F1 - 8 * (term_E0E1_F1 + term_E1E0_F1 + term_E0E1E0) + 4 * (term_E0E1 + term_E1E0 + term_E1_F1) \
            - 2 * (term_E1 + term_F1) + term_I

d_A0A1A0B0 = 16 * term_E0E1E0_F0 - 8 * (term_E0E1_F0 + term_E1E0_F0 + term_E0E1E0) + 4 * (term_E0E1 + term_E1E0 + term_E1_F0) \
            - 2 * (term_E1 + term_F0) + term_I

d_A1B0B1B0 = 16 * term_E1_F0F1F0 - 8 * (term_F0F1F0 + term_E1_F0F1 + term_E1_F1F0) + 4 * (term_E1_F1 + term_F0F1 + term_F1F0) \
            - 2 * (term_E1 + term_F1) + term_I

d_A0B0B1B0 = 16 * term_E0_F0F1F0 - 8 * (term_F0F1F0 + term_E0_F0F1 + term_E0_F1F0) + 4 * (term_E0_F1 + term_F0F1 + term_F1F0) \
            - 2 * (term_E0 + term_F1) + term_I

# 4-body terms (Mixed 2-2 split)
d_A0A1B0B1 = 16 * term_E0E1_F0F1 - 8 * (term_E0E1_F0 + term_E0E1_F1 + term_E0_F0F1 + term_E1_F0F1) \
            + 4 * (term_E0_F0 + term_E0_F1 + term_E1_F0 + term_E1_F1) \
            + 4 * (term_E0E1 + term_F0F1) \
            - 2 * (term_E0 + term_E1 + term_F0 + term_F1) \
            + term_I

d_A0A1B1B0 = 16 * term_E0E1_F1F0 - 8 * (term_E0E1_F1 + term_E0E1_F0 + term_E0_F1F0 + term_E1_F1F0) \
            + 4 * (term_E0_F1 + term_E0_F0 + term_E1_F1 + term_E1_F0) \
            + 4 * (term_E0E1 + term_F1F0) \
            - 2 * (term_E0 + term_E1 + term_F1 + term_F0) \
            + term_I

d_A1A0B0B1 = 16 * term_E1E0_F0F1 - 8 * (term_E1E0_F0 + term_E1E0_F1 + term_E1_F0F1 + term_E0_F0F1) \
            + 4 * (term_E1_F0 + term_E1_F1 + term_E0_F0 + term_E0_F1) \
            + 4 * (term_E1E0 + term_F0F1) \
            - 2 * (term_E1 + term_E0 + term_F0 + term_F1) \
            + term_I

d_A1A0B1B0 = 16 * term_E1E0_F1F0 - 8 * (term_E1E0_F1 + term_E1E0_F0 + term_E1_F1F0 + term_E0_F1F0) \
            + 4 * (term_E1_F1 + term_E1_F0 + term_E0_F1 + term_E0_F0) \
            + 4 * (term_E1E0 + term_F1F0) \
            - 2 * (term_E1 + term_E0 + term_F1 + term_F0) \
            + term_I

# 6-body term
d_A0A1A0B0B1B0 = 64 * term_E0E1E0_F0F1F0 \
            - 32 * (term_E0E1E0_F0F1 + term_E0E1E0_F1F0 + term_E0E1_F0F1F0 + term_E1E0_F0F1F0) \
            + 16 * (term_E0E1E0_F1 + term_E1_F0F1F0 + term_E0E1_F0F1 + term_E0E1_F1F0 + term_E1E0_F0F1 + term_E1E0_F1F0) \
            - 8 * (term_E0E1E0 + term_F0F1F0 + term_E0E1_F1 + term_E1E0_F1 + term_E1_F0F1 + term_E1_F1F0) \
            + 4 * (term_E0E1 + term_E1E0 + term_F0F1 + term_F1F0 + term_E1_F1) \
            - 2 * (term_E1 + term_F1) \
            + term_I