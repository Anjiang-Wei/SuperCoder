import requests
import argparse

from transformers import AutoTokenizer
import os
import ray
from datasets import load_dataset, Dataset
from sglang.utils import terminate_process, wait_for_server
from sglang.utils import launch_server_cmd
import tempfile
import subprocess
from tqdm import tqdm
import shutil
import psutil
import signal
import concurrent.futures
import json
import numpy as np
import fire
from pathlib import Path
from utils import is_same_assembly, run_with_timeout, compile_assembly, kill_process_tree
from test_benchmark import test_benchmark_generated_assembly
from metrics import calculate_metrics
from inference import *

# NEW: imports for explicit memory cleanup
import gc
try:
    import torch  # optional – only used if CUDA is available
except ImportError:
    torch = None



def merge_results(gen_results):
    # Flatten the list of lists while preserving sample_id
    gen_results = [item for sublist in gen_results for item in sublist]
    # Group by problem index
    problem_samples = {}
    for item in gen_results:
        idx = item['idx']
        if idx not in problem_samples:
            problem_samples[idx] = []
        problem_samples[idx].append(item)

    # Sort by problem index to maintain order
    sorted_indices = sorted(problem_samples.keys())
    gen_results_by_problem = [problem_samples[idx] for idx in sorted_indices]

    return gen_results_by_problem





def main(args):
    # Determine whether the current inference engine relies on Ray
    use_ray = args.inference_engine == 'sglang'
    if use_ray and not ray.is_initialized():
        # Keep default settings; user can adjust via env vars if needed
        ray.init()

    if 'HF_CACHE' in os.environ:
        test_dataset = load_dataset(args.ds_name, split=args.split, cache_dir=os.environ['HF_CACHE'])
    else:
        test_dataset = load_dataset(args.ds_name, split=args.split)
    # for debugging
    # test_dataset = test_dataset.select(range(1))  # Only keep first 2 samples
    test_dataset = test_dataset.map(lambda example, idx: {"idx": idx}, with_indices=True)
    
    eval_type = 'c_only' if args.c_only_prompt else 'main'
    all_output_data = {}

    for i in range(args.num_iterations):

        if not args.use_gt:
            fewshot_dataset = load_dataset(args.ds_name, split='fewshot', cache_dir=os.environ['HF_CACHE'])
            fewshot_questions = [item['extra_info']['question'] for item in fewshot_dataset][:args.num_fs]
            fewshot_answers = [item['extra_info']['answer'] for item in fewshot_dataset][:args.num_fs]

            gen_results = eval(f"{args.inference_engine}_generate")(  # noqa: F821
                test_dataset,
                args.model_name,
                args.temperature,
                args.max_new_tokens,
                fewshot_questions,
                fewshot_answers,
                args.best_of,
                args.num_workers,
                eval_type,
            )
            # print(f"Raw generation results: {json.dumps(gen_results, indent=2)}")
            gen_results_by_problem = merge_results(gen_results)
            current_model_name = args.model_name
        else:
            # For ground truth, create a single sample per problem
            gen_results_by_problem = [[{'idx': item['idx'], 'sample_id': 0, 'output': item['extra_info']['answer']}]
                                      for item in test_dataset]
            current_model_name = 'gt'

        test_case_results, results_by_problem_and_sample, compilation_failures, identical_assembly_samples = (
            test_benchmark_generated_assembly(
                gen_results_by_problem,
                test_dataset,
                args.num_workers,
            )
        )

        output_data = calculate_metrics(
            current_model_name,
            args.ds_name,
            args.best_of,
            gen_results_by_problem,
            test_dataset,
            results_by_problem_and_sample,
            compilation_failures,
            identical_assembly_samples,
        )

        all_output_data[i] = output_data

        if args.use_gt:
            break

        # NEW: explicit cleanup to release memory each iteration
        del gen_results, gen_results_by_problem, test_case_results, results_by_problem_and_sample
        del compilation_failures, identical_assembly_samples, output_data
        gc.collect()
        if torch and torch.cuda.is_available():
            torch.cuda.empty_cache()

    
    res_dir = (
        f"results/{eval_type}/{args.ds_name.split('/')[-1]}_{args.split}/{current_model_name.split('/')[-1]}"
        f"/num_iterations_{args.num_iterations}/{args.num_fs}-shot/best_of_{args.best_of}"
    )
    Path(res_dir).mkdir(parents=True, exist_ok=True)

    with open(f"{res_dir}/problem_results.json", 'w') as f:
        json.dump(all_output_data, f, indent=2)
    
    print(f"Results saved in '{res_dir}/problem_results.json'")

    # NEW: shut down Ray at the very end to free object store memory
    if use_ray and ray.is_initialized():
        ray.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ds_name", type=str, default='LLM4Code/llm_superoptimizer_ds')
    parser.add_argument("--model_name", type=str, default='meta-llama/Llama-3.2-1B-Instruct')
    parser.add_argument("--split", type=str, default='val')
    parser.add_argument("--inference_engine", type=str, default='sglang')
    parser.add_argument("--num_iterations", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_new_tokens", type=int, default=2000)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--use_gt", type=bool, default=False)
    parser.add_argument("--num_fs", type=int, default=0)
    parser.add_argument("--best_of", type=int, default=1)
    parser.add_argument("--c_only_prompt", type=bool, default=False)
    args = parser.parse_args()

    main(args)
