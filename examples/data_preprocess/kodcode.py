"""
Preprocess KodCode problems to parquet format.
KodCode-V1: https://huggingface.co/datasets/KodCode/KodCode-V1
KodCode-Light-RL-10K: https://huggingface.co/datasets/KodCode/KodCode-Light-RL-10K
"""

import os
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from datasets import load_dataset, concatenate_datasets
from rich.rule import Rule
import rich

from verl.utils.hdfs_io import copy, makedirs
from verl.utils.reward_score.coder1 import code_exec, remote_check_stdio, _ERROR_MSG_PREFIX

N_TESTSET_PER_DATASET = 512  # per dataset
_EMPTY_RETURN_ = {
    "data_source": None,
    "prompt": None,
    "ability": None,
    "reward_model": None,
    "extra_info": None,
}

SYSTEM_PROMPT = """You are a helpful programming assistant. \
The user will ask you a question and you as the assistant solve it. \
The assistant first thinks how to solve the task through reasoning and then provides the user with the final answer. \
The reasoning process and answer are enclosed within <think>...</think> and <answer>...</answer> tags, respectively."""

PY_IMPORTS = "import heapq\nfrom math import floor, gcd\nimport random\nimport sys\nfrom typing import *\nfrom functools import *\nimport collections\nfrom collections import *\nfrom itertools import *\nfrom heapq import *\nfrom bisect import *\nfrom string import *\nimport math\nimport datetime\ninf = float('inf')\n"

def kodcode():  
    rich.print(Rule("Loading KodCode..."))
    dataset = load_dataset("KodCode/KodCode-Light-RL-10K")

    packages = [
        "beautifulsoup4", "fake-useragent", "imageio", "keras", "lxml", "matplotlib", "numpy", "opencv-python",
        "pillow", "requests", "rich", "scikit-learn", "sphinx-pyproject", "statsmodels", "sympy", "tweepy",
        "typing_extensions", "xgboost", "flask", "seaborn"
    ]
    block_libs = [
        "fake-useragent", "keras", "socket", "torch", "scipy", "sklearn", "cv2", "scipy", "imageio", "sphinx-pyproject",
        "xgboost", "tweepy", "flask", "matplotlib", "pillow", "seaborn", "smtpd", "pandas", "bs4"
    ]

    def make_map_fn(split):

        def process_fn(example, idx):
            reference_solution = example["solution"]
            test_code = "from solution import *\n" + example["test"].strip()
            # skip it if reference solution requires libs from block_libs
            if any(lib in reference_solution for lib in block_libs):
                return _EMPTY_RETURN_
            if any(lib in test_code for lib in block_libs):
                return _EMPTY_RETURN_
            prompt = f"Please solve the programming task below in Python. \n\n{example['question'].strip()}"
            test_declaration = example["test_info"][0]["function_declaration"].strip()
            if test_declaration and test_declaration.strip():
                prompt += f"\n\nNote that the function declaration is {test_declaration}. Your code should be wrapped in a markdown code block."

            try:
                succ, err = code_exec(code=reference_solution, pytest=test_code)
                if not succ:
                    rich.print(f"[bold red]Test code failed for {example['question_id']}")
                    print(reference_solution)
                    print(err)
                    return _EMPTY_RETURN_
            except Exception as e:
                rich.print(f"[bold red]Exception during code execution for {example['question_id']}: {str(e)}")
                return _EMPTY_RETURN_

            return {
                "data_source": "code",
                "prompt": [
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT
                    },
                    {
                        "role": "user",
                        "content": prompt
                    },
                ],
                "ability": "coding",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": json.dumps({"pytest": test_code}),
                },
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "reference": reference_solution,
                    "prompt": prompt,
                    "dataset": "KodCode/KodCode-Light-RL-10K",
                },
            }

        return process_fn

    dataset = dataset.map(
        function=make_map_fn("train"),
        with_indices=True,
        num_proc=64,
    ).filter(lambda x: x != _EMPTY_RETURN_)
    splits = dataset['train'].shuffle(seed=666).filter(lambda x: x['prompt'] is not None).train_test_split(test_size=N_TESTSET_PER_DATASET, seed=666)
    
    train_dataset = splits["train"]
    test_dataset = splits["test"]
    return train_dataset, test_dataset


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", default="./data/")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    root_dir = args.root_dir
    hdfs_dir = args.hdfs_dir

    train_datasets = []
    test_datasets = []

    dataset_makes = [kodcode]

    for train, test in [make() for make in dataset_makes]:
        train_datasets.append(train)
        test_datasets.append(test)

    train_dataset = concatenate_datasets(train_datasets).shuffle(seed=666)
    test_dataset = concatenate_datasets(test_datasets)

    rich.print(Rule("Saving the final dataset"))
    print("Train set:", train_dataset)
    print("Test set:", test_dataset)

    local_dir = os.path.join(root_dir, f"kodcode-light-rl-10k")
    rich.print(f"[bold green]Saving to {local_dir}...")
    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=root_dir, dst=hdfs_dir)
