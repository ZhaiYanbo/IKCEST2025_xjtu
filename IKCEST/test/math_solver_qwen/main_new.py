from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import os
from tqdm import tqdm
import json
import time
import re
import torch
import sys
import re
import math
import fractions
from decimal import Decimal, InvalidOperation

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

choose_prompt = "You are a math teacher and now you need to complete a math multiple-choice question. Please put The thinking process in <think> thinking process </think>, and put the final chosen option in <answer>The answer is \\boxed{capitalized option}.</answer>."

fill_blank_prompt = "You are a math teacher and now you need to complete a math fill-in-the-blank question. Please put The thinking process in <think> thinking process </think>, and put the final filled result in <answer>The answer is \\boxed{filled result(rounded to 6 decimal places)}.</answer>."

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

def load_model(model_name="Qwen/Qwen2.5-VL-3B-Instruct"):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="cuda"
    )

    min_pixels = 256*28*28
    max_pixels = 1280*28*28
    processor = AutoProcessor.from_pretrained(model_name, min_pixels=min_pixels, max_pixels=max_pixels)
    model.eval()
    return model, processor

def load_jsonl(input_file):
    """加载jsonl文件"""
    with open(input_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    return data

def main(image_dir, input_jsonl, output_jsonl):
    # 记录总开始时间
    total_start_time = time.time()
    
    # 1. 模型加载阶段
    model, processor = load_model('/gemini/user/private/table_reasoning/zhaiyanbo/math/ms-swift-main/output/v13-20250730-094235/checkpoint-200')

    # 2. 数据加载阶段
    input_file = load_jsonl(input_jsonl)

    res = []
    processing_times = []  # 记录每个样本的处理时间
    
    # 3. 处理阶段
    for obj in tqdm(input_file, desc="Processing"):
        sample_start = time.time()
        
        image_path = os.path.join(image_dir, obj['image'])

        # 根据标签选择提示词
        tag = obj['tag']
        if tag == "选择题":
            system_prompt = choose_prompt
        elif tag == "填空题":
            system_prompt = fill_blank_prompt
        elif tag == "计算应用题":
            system_prompt = calculate_prompt

        # 准备模型输入
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

        # 预处理
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        
        # 模型推理
        inference_start = time.time()
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        # with torch.inference_mode():
        #     generated_ids = model.generate(**inputs, max_new_tokens=2048)
        # generated_ids_trimmed = [
        #     out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        # ]
        # output_text = processor.batch_decode(
        #     generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        # )[0]

        with torch.inference_mode():
            generated_ids = model.generate(**inputs, max_new_tokens=1024)
            
        # 计算新增 token 数量
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        # 获取第一个输出的新增 token 数量
        new_token_count = len(generated_ids_trimmed[0])

        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        # 添加结果输出
        print(f"输出文本共生成 {new_token_count} 个新 token")
        print("输出内容：", output_text)
        inference_time = time.time() - inference_start

        # 结果处理
        step, answer = safe_extract(output_text, tag)
        obj["steps"] = step
        obj["answer"] = answer
        res.append(obj)
        
        # 记录单个样本处理时间
        sample_time = time.time() - sample_start
        processing_times.append(sample_time)
        print(f"处理完成: {obj['image']} | 推理耗时: {inference_time:.2f}s | 总耗时: {sample_time:.2f}s")

    # 4. 结果保存阶段
    save_start = time.time()
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for item in res:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    save_time = time.time() - save_start

    # 5. 统计信息
    total_time = time.time() - total_start_time
    avg_process_time = sum(processing_times) / len(processing_times)
    
    stats = [
        "\n===== 运行统计 =====",
        f"总样本数: {len(input_file)}",
        f"平均单样本处理时间: {avg_process_time:.2f}秒",
        f"最长单样本处理时间: {max(processing_times):.2f}秒",
        f"最短单样本处理时间: {min(processing_times):.2f}秒",
        f"结果保存耗时: {save_time:.2f}秒",
        f"总运行时间: {total_time:.2f}秒",
        f"平均处理速度: {len(input_file)/total_time:.2f} 样本/秒"
    ]
    
    # 输出到控制台
    for stat in stats:
        print(stat)
    
    # 保存到txt文件
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    stats_filename = f"./reports/runtime_stats_{timestamp}.txt"
    with open(stats_filename, 'w', encoding='utf-8') as f:
        f.write("\n".join(stats))


def process_and_calculate_accuracy(input_jsonl, result_json):
    """处理数据并计算多种准确率"""
    # 读取所有数据
    res = []
    with open(input_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            res.append(json.loads(line))
    
    # 初始化统计变量
    stats = {
        "total": {"count": 0, "correct": 0},
        "example": {"count": 0, "correct": 0},
        "test": {"count": 0, "correct": 0}
    }
    
    processed_data = []
    
    for obj in res:

        is_correct = obj['answer'] == obj['groundtruth']
        obj["correct"] = is_correct
        
        # 更新统计信息
        stats["total"]["count"] += 1
        if is_correct:
            stats["total"]["correct"] += 1
        
        # 检查ID包含"example"的情况
        if "example" in obj["id"].lower():
            stats["example"]["count"] += 1
            if is_correct:
                stats["example"]["correct"] += 1
        
        # 检查ID包含"test"的情况
        if "test" in obj["id"].lower():
            stats["test"]["count"] += 1
            if is_correct:
                stats["test"]["correct"] += 1
        
        processed_data.append(obj)
    
    results = {
        "overall_accuracy": stats["total"]["correct"] / stats["total"]["count"] if stats["total"]["count"] > 0 else 0,
        "example_accuracy": stats["example"]["correct"] / stats["example"]["count"] if stats["example"]["count"] > 0 else 0,
        "test_accuracy": stats["test"]["correct"] / stats["test"]["count"] if stats["test"]["count"] > 0 else 0,
        "counts": {
            "total": stats["total"]["count"],
            "example": stats["example"]["count"],
            "test": stats["test"]["count"]
        },
        "correct_counts": {
            "total": stats["total"]["correct"],
            "example": stats["example"]["correct"],
            "test": stats["test"]["correct"]
        },
        "details": [{
            "id": obj["id"],
            "correct": obj["correct"],
            "type": "example" if "example" in obj["id"].lower() else ("test" if "test" in obj["id"].lower() else "other")
        } for obj in processed_data]
    }
    
    # 保存统计结果
    with open(result_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    return results

if __name__ == "__main__":
    # main("../sample_output", "../sample_output/test.jsonl", "../sample_output/results/test_grpo_new.jsonl")
    # process_and_calculate_accuracy("../sample_output/results/test_grpo_new.jsonl", "./reports/grpo_results_new.json")
    process_and_calculate_accuracy ("/gemini/user/private/table_reasoning/zhaiyanbo/verl/IKCEST/submission/output/dapo_result_0809.jsonl","/gemini/user/private/table_reasoning/zhaiyanbo/verl/IKCEST/submission/output/report/dapo_result_0809.jsonl")