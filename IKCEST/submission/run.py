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

MODEL_PATH = '/gemini/platform/public/NLP/model/project_model/math/grpo_0805/global_step_100/merged_hf_model'

choose_prompt = "You are a math teacher and now you need to complete a math multiple-choice question. Please put The thinking process in <think> thinking process </think>, and put the final chosen option in <answer>The answer is \\boxed{capitalized option}.</answer>."

# fill_blank_prompt = "You are a math teacher and now you need to complete a math fill-in-the-blank question. Please put The thinking process in <think> thinking process </think>, and put the final filled result in <answer>The answer is \\boxed{filled result(rounded to 6 decimal places)}.</answer>."

calculate_prompt = "You are a math teacher and now you need to complete a math calculation application problem. Please put The thinking process into <think> thinking process </think>, and put the final calculation result into <answer>The answer is \\boxed{calculation result(rounded to 6 decimal places)}.</answer>."

initial_prompt="""<image>\nSolve the problem in the image."""


def extract_steps_and_answer(response, question_type):
    """
    从模型响应中提取 <think> 步骤内容和 <answer> 中的答案
    新格式: <think>steps字段</think>\n<answer>The answer is \\boxed{原jsonl中的answer字段}.</answer>
    
    更新：改进 \\boxed{} 提取逻辑，处理嵌套表达式
    """
    # 定义正则表达式模式（改进版）
    think_pattern = r"<think>(.*?)</think>"
    answer_tag_pattern = r"<answer>(.*?)</answer>"
    
    # 匹配 <think> 中的步骤内容
    think_match = re.search(think_pattern, response, re.DOTALL)
    step = think_match.group(1).strip() if think_match else ""
    
    # 1. 提取 <answer> 标签内容
    answer_tag_match = re.search(answer_tag_pattern, response, re.DOTALL)
    answer_tag_content = answer_tag_match.group(1).strip() if answer_tag_match else ""
    
    # 2. 使用平衡括号匹配提取 \\boxed{} 内容
    def extract_boxed_with_balance(s):
        """提取平衡括号的 \\boxed{...} 内容"""
        start = s.find(r'\boxed{')
        if start == -1:
            return None
        
        # 从第一个左括号开始计数
        start_idx = start + len(r'\boxed{')
        count = 1
        current = start_idx
        
        while current < len(s) and count > 0:
            if s[current] == '{':
                count += 1
            elif s[current] == '}':
                count -= 1
            current += 1
        
        if count == 0:
            return s[start_idx:current-1]  # 去掉最后的右括号
        return None
    
    # 3. 从 <answer> 中提取答案2
    answer2 = extract_boxed_with_balance(answer_tag_content) if answer_tag_content else ""
    
    # 4. 从 <think> 中提取最后一个答案1（处理嵌套表达式）
    all_boxed_in_think = []
    if think_match:
        # 提取所有平衡的 \\boxed{} 内容
        content = think_match.group(1)
        pos = 0
        while True:
            match = extract_boxed_with_balance(content[pos:])
            if not match:
                break
            all_boxed_in_think.append(match)
            pos = content.find(r'\boxed{', pos + 1)  # 移动到下一个位置
    
    # 取最后一个 \\boxed{} 作为答案1
    answer1 = all_boxed_in_think[-1] if all_boxed_in_think else ""
    
    # 5. 答案处理逻辑（保留小数点后6位）
    def format_answer(ans):
        """格式化答案为小数点后6位（保留有效数字）"""
        if not ans:
            return "1.000000"
        
        try:
            # 处理分数格式（latex和文本格式）
            if r'\frac' in ans or '/' in ans:
                # 提取分子和分母
                if r'\frac' in ans:
                    num_match = re.search(r'\\frac\{(\d+)\}\{(\d+)\}', ans)
                    if num_match:
                        numerator = int(num_match.group(1))
                        denominator = int(num_match.group(2))
                        result = numerator / denominator
                    else:
                        # 尝试其他分数格式
                        num_match = re.search(r'\\frac\{([^}]+)\}\{([^}]+)\}', ans)
                        if num_match and num_match.group(1).isdigit() and num_match.group(2).isdigit():
                            numerator = int(num_match.group(1))
                            denominator = int(num_match.group(2))
                            result = numerator / denominator
                        else:
                            return "1.000000"
                else:
                    # 处理文本分数格式：1/2, 3/4
                    if '/' in ans:
                        parts = ans.split('/')
                        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                            result = int(parts[0]) / int(parts[1])
                        else:
                            return "1.000000"
            else:
                # 直接数值处理
                result = float(ans)
                
            # 格式化到6位小数
            if result.is_integer():
                return f"{int(result):d}.000000"
            return f"{result:.6f}".rstrip('0').rstrip('.') + ('0'*(6 - len(f"{result:.6f}".rstrip('0').rstrip('.').split('.')[1])))
        
        except (ValueError, TypeError, InvalidOperation, ZeroDivisionError):
            return "1.000000"
    
    # 6. 根据问题类型选择最终答案
    if question_type == "选择题":
        return step, answer2
    else:
        ans1 = format_answer(answer1)
        ans2 = format_answer(answer2)
        
        # 尝试不同精度比较
        try:
            if math.isclose(float(ans1), float(ans2), rel_tol=1e-3):
                return step, ans2
            return step, ans1
        except:
            return step, "1.000000"

def safe_extract(response, question_type):
    try:
        return extract_steps_and_answer(response, question_type)
    except:
        if question_type == "选择题":
            return "", "C"
        else:
            return "", "1.000000"

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
            # system_prompt = fill_blank_prompt
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
        print(output_text)
        
        step, answer = safe_extract(output_text, tag)
        obj["steps"] = step
        obj["answer"] = answer
        obj['output_text'] = output_text
        res.append(obj)
    
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for item in res:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


if __name__ == "__main__":
    # main(sys.argv[1], sys.argv[2], sys.argv[3])
    main('/gemini/user/private/table_reasoning/zhaiyanbo/math/test/sample_output','/gemini/user/private/table_reasoning/zhaiyanbo/math/test/sample_output/test.jsonl','./output/grpo_result_0804.jsonl')