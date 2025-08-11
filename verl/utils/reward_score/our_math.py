import re


def compute_score(
    solution_str,
    ground_truth,
    method="strict",
    structure_weight=0.5,
    boxed_weight=0.5,
    correct_weight=1.0,
):
    """
    返回结构匹配得分、boxed格式匹配得分、最终答案匹配得分，以及总得分。
    """
    s = solution_str.strip()
    structure_score = -0.5
    boxed_score = 0.0
    correct_score = 0.0

    # ===== 第一步：结构匹配 =====
    structure_gate_pattern = re.compile(
        r'^<think>.*?</think>\n<answer>.*?\\boxed\{.*?\}.*?</answer>\s*$',
        re.DOTALL,
    )
    if structure_gate_pattern.match(s):
        structure_score = structure_weight

        # ===== 第二步：boxed 内容匹配（六位小数或 A-F） =====
        answer_extract_pattern = re.compile(r'<answer>(.*?)</answer>', re.DOTALL)
        boxed_pattern = re.compile(r'\\boxed\{((?:[^{}]*|\{.*?\})*)\}', re.DOTALL)
        float6_pattern = re.compile(r'^[+-]?(\d+)(\.\d{6})$')
        choice_core = r'[A-F]'
        choice_pattern = re.compile(r'^[A-F]$')

        boxed_ok = False
        extracted = None
        ans_match = answer_extract_pattern.search(s)
        if ans_match:
            ans_content = ans_match.group(1)
            m = boxed_pattern.search(ans_content)
            if m:
                raw = m.group(1).strip().replace(" ", "")
                if float6_pattern.match(raw):
                    boxed_ok, extracted = True, raw
                elif choice_pattern.match(raw):
                    boxed_ok, extracted = True, raw

        if boxed_ok:
            boxed_score = boxed_weight

            # ===== 第三步：值是否正确 =====
            if extracted == ground_truth:
                correct_score = correct_weight

    # ===== 总得分 =====
    total_score = structure_score + boxed_score + correct_score
    acc = total_score == 2
    return {'score': total_score, 'structure':structure_score, 'boxed':boxed_score,'correct':correct_score,'acc':acc}    