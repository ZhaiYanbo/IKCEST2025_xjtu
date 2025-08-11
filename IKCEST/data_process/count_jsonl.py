import json, re, sys
from collections import Counter
from typing import Dict, Any, List

CHOICE_RE = re.compile(r"\b([A-Da-d])\b")
BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")
SIX_DECIMAL_RE = re.compile(r"^[+-]?\d+\.\d{6}$")  # 严格六位小数格式

def normalize_choice_answer(ans: Any) -> str:
    """将 answer 规范化为 A/B/C/D/UNKNOWN/OTHER，用于 multi-choice 计数。"""
    if ans is None:
        return "UNKNOWN"
    s = str(ans).strip()

    # 优先 \boxed{...}
    m = BOXED_RE.search(s)
    if m:
        s = m.group(1).strip()

    # 抽取单个字母选项
    m = CHOICE_RE.search(s)
    if m:
        return m.group(1).upper()

    # 兜底：过长则 OTHER，短文本直接返回
    s = re.sub(r"\s+", " ", s)
    return s if len(s) <= 10 else "OTHER"

def is_six_decimal_format(ans: Any) -> bool:
    """严格校验是否为六位小数字符串格式：^[+-]?digits.d{6}$"""
    if ans is None:
        return False
    s = str(ans).strip()

    # 若出现 \boxed{...}，先取里面
    m = BOXED_RE.search(s)
    if m:
        s = m.group(1).strip()

    # 去掉可能的千分位逗号（如 1,234.123456）
    s = s.replace(",", "")
    return bool(SIX_DECIMAL_RE.fullmatch(s))

def analyze_jsonl(in_path: str):
    mc_counter = Counter()
    invalid_non_mc: List[Dict[str, Any]] = []

    with open(in_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                invalid_non_mc.append({"line_no": idx, "error": f"JSON decode error: {e}"})
                continue

            typ = obj.get("type")
            ans = obj.get("answer")

            if typ == "multi-choice":
                choice = normalize_choice_answer(ans)
                mc_counter[choice] += 1
            else:
                if not is_six_decimal_format(ans):
                    invalid_non_mc.append({
                        "line_no": idx,
                        "type": typ,
                        "answer": ans
                    })

    result = {
        "multi_choice_counts": dict(mc_counter),
        "invalid_non_multi_choice": invalid_non_mc
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python analyze.py input.jsonl")
        sys.exit(1)
    analyze_jsonl(sys.argv[1])