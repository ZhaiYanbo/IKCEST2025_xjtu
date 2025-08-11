# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the our math rl dataset to parquet format
"""

import argparse
import os

import datasets

from verl.utils.hdfs_io import copy, makedirs


choose_prompt = "You are a math teacher and now you need to complete a math multiple-choice question. Please put The thinking process in <think> thinking process </think>, and put the final chosen option in <answer>The answer is \\boxed{capitalized option}.</answer>."

# fill_blank_prompt = "You are a math teacher and now you need to complete a math fill-in-the-blank question. Please put The thinking process in <think> thinking process </think>, and put the final filled result in <answer>The answer is \\boxed{filled result(rounded to 6 decimal places)}.</answer>."

calculate_prompt = "You are a math teacher and now you need to complete a math calculation application problem. Please put The thinking process into <think> thinking process </think>, and put the final calculation result into <answer>The answer is \\boxed{calculation result(rounded to 6 decimal places)}.</answer>."

initial_prompt="""<image>\nSolve the problem in the image."""

def norm_images(imgs):
    if imgs is None: return []
    if isinstance(imgs, (str, dict)): imgs = [imgs]
    out = []
    for im in imgs:
        if isinstance(im, str): out.append({"image_url": im})
        elif isinstance(im, dict):
            if "image_url" in im: out.append(im)
            elif "path" in im:    out.append({"image_url": im["path"]})
            elif "url" in im:     out.append({"image_url": im["url"]})
            else: raise ValueError(f"Unsupported image spec: {im}")
        else:
            out.append({"image": im})  # 视底层是否支持
    return out

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="/gemini/user/private/table_reasoning/zhaiyanbo/verl/data/rl_data")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--data_dir", default="/gemini/user/private/table_reasoning/zhaiyanbo/math/dataset_grpo", help="包含本地 JSONL 的目录")


    args = parser.parse_args()

    data_files = {
        "train": os.path.join(args.data_dir, "grpo_convert_train.jsonl"),
        "test":  os.path.join(args.data_dir, "grpo_convert_test.jsonl"),
    }

    data_source = "grpo_convert"

    # dataset = datasets.load_dataset(data_source)

    dataset = datasets.load_dataset("json", data_files=data_files, split=None, keep_in_memory=False)

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    
    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            problem = example.pop("problem")
            instruction_following = example.pop("instrution")
            prompt = instruction_following + "\n" + problem
            answer = example.pop("answer")
            images = example.pop("image")

            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                "images": norm_images(images),
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": answer},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": answer,
                    "question": problem,
                },
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True, num_proc=8)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True, num_proc=8)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
