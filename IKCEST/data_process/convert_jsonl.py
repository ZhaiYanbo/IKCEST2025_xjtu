import json, re, sys

def extract_instruction(obj):
    try:
        content = obj["messages"][0]["content"]
        if content == "You are a math teacher and now you need to complete a math fill-in-the-blank question. Please put The thinking process in <think> thinking process </think>, and put the final filled result in <answer>The answer is \\boxed{filled result (rounded to 6 decimal places)}.</answer>.":
            content = "You are a math teacher and now you need to complete a math calculation application problem. Please put The thinking process into <think> thinking process </think>, and put the final calculation result into <answer>The answer is \\boxed{calculation result(rounded to 6 decimal places)}.</answer>."
        return content
    except Exception:
        return ""
    
def extract_type(obj):
    type = ""
    try:
        content = obj["messages"][0]["content"]
        if content == "You are a math teacher and now you need to complete a math fill-in-the-blank question. Please put The thinking process in <think> thinking process </think>, and put the final filled result in <answer>The answer is \\boxed{filled result (rounded to 6 decimal places)}.</answer>.":
            type = "fill-in-blank"
        elif content == "You are a math teacher and now you need to complete a math multiple-choice question. Please put The thinking process in <think> thinking process </think>, and put the final chosen option in <answer>The answer is \\boxed{capitalized option}.</answer>.":
            type = "multi-choice"
        else:
            print(content)
    except Exception:
        return ""
    return type

def extract_problem(obj):
    content = "<image>\nSolve the problem in the image."
    
    return content.strip()

def extract_answer(obj):
    solutions = obj.get("solutions", "")
    if not isinstance(solutions, str):
        solutions = str(solutions)

    # 1) 优先从 \boxed{...} 中抽取
    m = re.search(r"\\boxed\{([^{}]+)\}", solutions)
    if m:
        return m.group(1).strip()

    # 2) 次选：从 <answer>...</answer> 中抽取
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", solutions, flags=re.S | re.I)
    if m:
        # 去掉可能的 “The answer is \boxed{X}.” 等包装
        inner = m.group(1).strip()
        mm = re.search(r"\\boxed\{([^{}]+)\}", inner)
        return mm.group(1).strip() if mm else inner

    # 3) 再兜底：寻找 “answer is ...” 的句式
    m = re.search(r"answer\s+is\s+([A-D]|[^\.\n]+)", solutions, flags=re.I)
    return m.group(1).strip() if m else ""

def extract_image(obj):
    imgs = obj.get("images", [])
    if isinstance(imgs, list) and imgs:
        return imgs[0]          # 若想保留列表，改为：return imgs
    return imgs if isinstance(imgs, (str, list)) else ""

def transform_line(line):
    obj = json.loads(line)
    out = {
        "instrution": extract_instruction(obj),   # 按你写法用 instrution（非 instruction）
        "problem": extract_problem(obj),
        "answer": extract_answer(obj),
        "image": extract_image(obj),
        "type": extract_type(obj)
    }
    return json.dumps(out, ensure_ascii=False)

def main(in_path, out_path):
    with open(in_path, "r", encoding="utf-8") as fin, \
         open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                fout.write(transform_line(line) + "\n")
            except Exception as e:
                # 出错行可按需记录或跳过
                sys.stderr.write(f"[WARN] Skip line due to error: {e}\n")

if __name__ == "__main__":
    # 用法：python convert_jsonl.py input.jsonl output.jsonl
    if len(sys.argv) != 3:
        print("Usage: python convert_jsonl.py input.jsonl output.jsonl")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])