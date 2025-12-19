import json
import os


def split_projector_term(term):
    """
    核心拆分函数：将项一分为二。

    通用逻辑：
    - 左侧为 Alice (E) 部分，右侧为 Bob (F) 部分。
    - 例如: E0 E1 F0 -> (E0 E1, F0)

    特殊逻辑 (针对 Bob 方长度为 4 的交替项):
    - 如果 Bob 部分是 F0 F1 F0 F1 或 F1 F0 F1 F0
    - 拆分方式改为：
      左侧 = Alice部分 + Bob前半部分
      右侧 = Bob后半部分
    - 例如: E0 F0 F1 F0 F1 -> (E0 F0 F1, F0 F1)
    """
    tokens = term.split()

    # 筛选以 'E' 开头的算子 (Alice)
    alice_tokens = [t for t in tokens if t.startswith('E')]
    # 筛选以 'F' 开头的算子 (Bob)
    bob_tokens = [t for t in tokens if t.startswith('F')]

    # --- 特殊逻辑检测 ---
    # 检测 Bob 部分是否为长度为 4 的特定交替序列
    is_special_bob = False
    if len(bob_tokens) == 4:
        # 检查是否为 F0 F1 F0 F1 或 F1 F0 F1 F0
        bob_str_check = " ".join(bob_tokens)
        if bob_str_check in ["F0 F1 F0 F1", "F1 F0 F1 F0"]:
            is_special_bob = True

    if is_special_bob:
        # 特殊拆分：Bob 分两半
        # 前半部分 (2个)
        bob_half_1 = bob_tokens[:2]
        # 后半部分 (2个)
        bob_half_2 = bob_tokens[2:]

        # 左侧 = Alice + Bob前半
        left_tokens = alice_tokens + bob_half_1
        # 右侧 = Bob后半
        right_tokens = bob_half_2

        left_str = " ".join(left_tokens) if left_tokens else "I"
        right_str = " ".join(right_tokens) if right_tokens else "I"

        return left_str, right_str

    else:
        # --- 通用逻辑 ---
        # Alice 永远在左侧
        alice_str = " ".join(alice_tokens) if alice_tokens else "I"
        # Bob 永远在右侧
        bob_str = " ".join(bob_tokens) if bob_tokens else "I"

        return alice_str, bob_str


def analyze_and_split(data):
    # 用于存储分类结果
    classified_terms = {1: [], 2: [], 3: [], 4: [], 5: [], 6: []}

    print(f"{'Term':<30} | {'Length':<6} | {'Split (Left part, Right part)'}")
    print("-" * 75)

    for term in data.keys():
        # 1. 识别长度
        tokens = term.split()
        length = len(tokens)

        # 记录分类
        if length in classified_terms:
            classified_terms[length].append(term)
        elif length > 6:
            # 处理可能的超长项
            if 'other' not in classified_terms: classified_terms['other'] = []
            classified_terms['other'].append(term)

        # 2. 自动拆分逻辑 (应用于所有项)
        left_part, right_part = split_projector_term(term)
        print(f"{term:<30} | {length:<6} | ('{left_part}', '{right_part}')")

    print("\n" + "=" * 30)
    print("分类统计 (Count per length):")
    # 按顺序输出统计结果
    sorted_keys = sorted([k for k in classified_terms.keys() if isinstance(k, int)])
    for length in sorted_keys:
        print(f"Length {length}: {len(classified_terms[length])} items")

    if 'other' in classified_terms:
        print(f"Length >6: {len(classified_terms['other'])} items")


def main():
    # 1. 获取当前脚本所在的目录 (即 test_fun 文件夹)
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        # 兼容某些交互式环境
        current_dir = os.getcwd()

    # 2. 构建指向 ../func/fidelity_coeffs.json 的路径
    # 使用 os.path.join 和 os.path.pardir (..) 来跨文件夹引用
    json_path = os.path.abspath(os.path.join(current_dir, '..', 'func', 'fidelity_coeffs.json'))

    # 3. 检查文件是否存在并读取
    if not os.path.exists(json_path):
        print(f"错误: 找不到文件 {json_path}")
        print("请确认文件夹结构为:")
        print("  Parent/")
        print("    func/fidelity_coeffs.json")
        print("    test_fun/split_projector.py")
        return

    print(f"正在读取文件: {json_path}\n")

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 4. 执行分析
        analyze_and_split(data)

    except json.JSONDecodeError:
        print("错误: JSON 文件格式不正确。")
    except Exception as e:
        print(f"发生未知错误: {e}")


if __name__ == "__main__":
    main()