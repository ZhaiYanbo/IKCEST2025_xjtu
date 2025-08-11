from vllm import SamplingParams, LLM
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
import os
from tqdm import tqdm
import json
import time
import re
import torch
import sys
import math
import fractions
from decimal import Decimal, InvalidOperation

MODEL_PATH = "/gemini/platform/public/NLP/private/project_model/math/ckpts/DAPO-Qwen2.5-VL-Math/global_step_180/merged_hf_model"

choose_prompt = "You are a math teacher and now you need to complete a math multiple-choice question. Please put The thinking process in <think> thinking process </think>, and put the final chosen option in <answer>The answer is \\boxed{capitalized option}.</answer>."

# fill_blank_prompt = "You are a math teacher and now you need to complete a math fill-in-the-blank question. Please put The thinking process in <think> thinking process </think>, and put the final filled result in <answer>The answer is \\boxed{filled result(rounded to 6 decimal places)}.</answer>."

calculate_prompt = "You are a math teacher and now you need to complete a math calculation application problem. Please put The thinking process into <think> thinking process </think>, and put the final calculation result into <answer>The answer is \\boxed{calculation result(rounded to 6 decimal places)}.</answer>."

initial_prompt="""<image>\nSolve the problem in the image."""

def extract_steps_and_answer(response, question_type):
    """
    从模型响应中提取 <think> 步骤内容和 <answer> 中的答案，结构清晰，鲁棒性强。
    """
    # 提取 <think> 和 <answer> 区块
    think_match = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
    answer_match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
    step = think_match.group(1).strip() if think_match else ""
    answer_text = answer_match.group(1).strip() if answer_match else ""

    # ---------- 工具函数 ----------
    def extract_boxed(s):
        start = s.find(r'\boxed{')
        if start == -1:
            return None
        start_idx = start + len(r'\boxed{')
        count = 1
        current = start_idx
        while current < len(s) and count > 0:
            if s[current] == '{':
                count += 1
            elif s[current] == '}':
                count -= 1
            current += 1
        return s[start_idx:current-1] if count == 0 else None

    def is_valid_letter_option(ans):
        return re.fullmatch(r'[A-Z]', ans) is not None
    def is_valid_6_digit_number(ans):
        return re.fullmatch(r'-?\d+\.\d{6}', ans) is not None

    def format_to_6_digits(ans):
        try:
            return f"{float(ans):.6f}"
        except:
            return "1.000000"

    def extract_last_uppercase_letter(s):
        matches = re.findall(r'\b[A-Z]\b', s)
        return matches[-1] if matches else None

    def extract_last_number(s):
        matches = re.findall(r'-?\d+(?:\.\d+)?', s)
        return matches[-1] if matches else None

    # ---------- 主要提取逻辑 ----------
    # 1. 先尝试从 <answer> 中提取 boxed 内容
    boxed_ans = extract_boxed(answer_text)

    if boxed_ans:
        if is_valid_letter_option(boxed_ans) or is_valid_6_digit_number(boxed_ans):
            return step, boxed_ans
        else:
            if question_type == "选择题":
                return step, "C"
            else:
                return step, format_to_6_digits(boxed_ans)

    # 2. 若 boxed 提取失败，根据题型 fallback 提取
    fallback_text = answer_text or step

    if question_type == "选择题":
        letter = extract_last_uppercase_letter(fallback_text)
        return step, letter if letter else "C"
    else:
        number = extract_last_number(fallback_text)
        return step, format_to_6_digits(number) if number else "1.000000"

def safe_extract(response, question_type):
    try:
        return extract_steps_and_answer(response, question_type)
    except:
        return ("", "C") if question_type == "选择题" else ("", "1.000000")

def load_jsonl(input_file):
    """加载jsonl文件"""
    with open(input_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    return data


def main(image_dir, input_jsonl, output_jsonl):
    sampling_params = SamplingParams(
        temperature=0,       
        max_tokens=4096,       
    )

    llm = LLM(
        model=MODEL_PATH,
        dtype="half",
        gpu_memory_utilization=0.8,     
        max_model_len=8192,           
        tensor_parallel_size=1          
    )

    min_pixels = 256*28*28
    max_pixels = 1280*28*28
    processor = AutoProcessor.from_pretrained(MODEL_PATH, min_pixels=min_pixels, max_pixels=max_pixels)

    input_file = load_jsonl(input_jsonl)

    res = []

    for obj in tqdm(input_file, desc="Processing"):
        
        image_path = os.path.join(image_dir, obj['image'])

        tag = obj['tag']
        if tag == "选择题":
            system_prompt = choose_prompt
        elif tag == "填空题":
            system_prompt = calculate_prompt
        elif tag == "计算应用题":
            system_prompt = calculate_prompt

        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": system_prompt},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": initial_prompt},
                ],
            }
        ]

        text_prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        mm_data = {}
        if image_inputs is not None:
            mm_data["image"] = image_inputs
        if video_inputs is not None:
            mm_data["video"] = video_inputs

        llm_inputs = {"prompt": text_prompt, "multi_modal_data": mm_data}

        outputs = llm.generate([llm_inputs], sampling_params=sampling_params)
        output_text = outputs[0].outputs[0].text.strip()

        step, answer = safe_extract(output_text, tag)
        obj["steps"] = step
        obj["answer"] = answer
        res.append(obj)
    
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for item in res:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])