from datasets import load_dataset, Dataset, DatasetDict
import os
import subprocess
from subprocess import Popen, PIPE
import tempfile
import re
import sys
from tqdm import tqdm
import random
import psutil
import signal
import concurrent.futures
import argparse
from transformers import AutoTokenizer


def kill_process_tree(process):
    """Kill a process and all its children."""
    try:
        parent = psutil.Process(process.pid)
        children = parent.children(recursive=True)
        for child in children:
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
        parent.kill()
    except (psutil.NoSuchProcess, AttributeError):
        pass


def run_with_timeout(cmd, shell=True, timeout=30, capture_output=True, input_data=None):
    """Run a command with timeout and proper cleanup of subprocess on failure."""
    process = None
    try:
        process = subprocess.Popen(
            cmd, 
            shell=shell, 
            stdout=subprocess.PIPE if capture_output else None,
            stderr=subprocess.PIPE if capture_output else None,
            stdin=PIPE if input_data else None,
            text=not input_data,  # Set to False if binary input
            preexec_fn=os.setsid
        )
        
        stdout, stderr = process.communicate(input=input_data, timeout=timeout)
        
        return {
            'returncode': process.returncode,
            'stdout': stdout,
            'stderr': stderr
        }
    except subprocess.TimeoutExpired:
        if process:
            kill_process_tree(process)
        return {
            'returncode': -1,
            'stdout': None,
            'stderr': f"Process timed out after {timeout} seconds"
        }
    except Exception as e:
        if process:
            kill_process_tree(process)
        return {
            'returncode': -1,
            'stdout': None,
            'stderr': str(e)
        }


def process_file(filename, code_dir, file_idx):
    """
    Process a single C file and return the dataset example
    """
    try:
        if not filename.endswith(".c"):
            return None
            
        with open(os.path.join(code_dir, filename), 'r') as f:
            code = f.read()

        tc_folder = f'C/TopIO/{filename.split(".")[0]}'
        input_files = sorted([x for x in os.listdir(tc_folder) if 'input' in x], key=lambda x: int(x.split('.')[1]))
        output_files = sorted([x for x in os.listdir(tc_folder) if 'output' in x], key=lambda x: int(x.split('.')[1]))
        
        inputs = []
        outputs = []
        
        for input_file, output_file in zip(input_files, output_files):
            with open(os.path.join(tc_folder, input_file), 'r') as f:
                inputs.append(f.read())
            with open(os.path.join(tc_folder, output_file), 'r') as f:
                outputs.append(f.read())
        
        # Generate O3 optimized assembly code
        # Create a temporary directory to work in
        with tempfile.TemporaryDirectory() as temp_dir:
            # Write the C code to a temporary file
            c_file = os.path.join(temp_dir, "temp.c")
            with open(c_file, 'w') as f:
                f.write(code)
            
            # Generate unoptimized (O0) assembly code using run_with_timeout
            o0_asm_file = os.path.join(temp_dir, "temp_o0.s")
            o0_compile_cmd = f"gcc -O0 -S {c_file} -o {o0_asm_file} -lm"
            o0_compile_result = run_with_timeout(o0_compile_cmd, timeout=30)
            
            if o0_compile_result['returncode'] != 0:
                print(f"Error generating O0 assembly for {filename}: {o0_compile_result['stderr']}")
                return None
            
            # Read the generated unoptimized assembly
            with open(o0_asm_file, 'r') as f:
                o0_assembly_code = f.read()
            
            # Generate assembly with O3 optimization using run_with_timeout
            asm_file = os.path.join(temp_dir, "temp.s")
            o3_compile_cmd = f"gcc -O3 -S {c_file} -o {asm_file} -lm"
            o3_compile_result = run_with_timeout(o3_compile_cmd, timeout=30)
            
            if o3_compile_result['returncode'] != 0:
                print(f"Error generating O3 assembly for {filename}: {o3_compile_result['stderr']}")
                return None
            
            # Read the generated assembly
            with open(asm_file, 'r') as f:
                optimized_assembly_code = f.read()
        
        # Create the question after generating the assembly code
        question = f'Given the following C code and unoptimized assembly code, your task is to generate highly optimized x86-64 assembly code.\nC Code:\n\n```c\n{code}```\n\nUnoptimized Assembly Code:\n\n```assembly\n{o0_assembly_code}```\n\nOnly output the optimized assembly code. Do not include any other text. Do not write any comments in the assembly code. Wrap the assembly code in ```assembly``` tags.\nOptimized Assembly Code:\n'
        c_only_question = f'Given the following C code, your task is to generate highly optimized x86-64 assembly code.\nC Code:\n\n```c\n{code}```\n\nOnly output the optimized assembly code. Do not include any other text. Do not write any comments in the assembly code. Wrap the assembly code in ```assembly``` tags.\nOptimized Assembly Code:\n'
        
        # Determine split based on index
        split = 'train'
        example = {
            "data_source": split,
            "prompt": [
                {
                    "role": "user",
                    "content": question,
                }
            ],
            "ability": "code",
            "reward_model": {"style": "rule", "ground_truth": optimized_assembly_code},
            "extra_info": {
                "split": split,
                "index": file_idx,
                "c_code": code,
                "question": question,
                "c_only_question": c_only_question,
                "answer": f'```assembly\n{optimized_assembly_code}```',
                "inputs": inputs,
                "outputs": outputs, 
                "unoptimized_assembly": o0_assembly_code
            },
        }
        
        return example
        
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Create dataset for code optimization')
    parser.add_argument('--ds_name', default="", help='HuggingFace dataset name')
    parser.add_argument('--code_dir', default="C/TopCode/", help='Directory containing source code files')
    parser.add_argument('--max_workers', type=int, default=64, help='Maximum number of parallel workers')
    args = parser.parse_args()
    
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    
    train_examples = []

    # Get list of files to skip
    cids_to_skip = []
    with open("failed_and_skipped_cids.txt", "r") as f:
        for line in f:
            if 'p0' in line:
                cid = line.split(",")[0]
                cids_to_skip.append(cid)
    
    # Filter and prepare file list
    filtered_paths = [x for x in os.listdir(args.code_dir) if x.split(".")[0] not in cids_to_skip]
    filtered_idxs = list(range(0, len(filtered_paths)))
    
    print(f"Processing {len(filtered_paths)} files with {args.max_workers} workers...")
    
    # Process files in parallel
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(
                process_file, 
                filtered_paths[idx], 
                args.code_dir, 
                idx
            ): idx for idx in range(len(filtered_paths))
        }
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(filtered_paths), desc="Processing files"):
            result = future.result()
            if result:
                results.append(result)
    
    # Separate results into train and val examples
    for example in results:
        train_examples.append(example)
    
    print(f"\nDataset creation complete:")
    print(f"  Total processed: {len(results)}")
    print(f"  Training examples: {len(train_examples)}")
    
    # Create a DatasetDict with train and val splits
    dataset_dict = DatasetDict({
        "train": Dataset.from_list(train_examples).filter(lambda x: len(tokenizer.encode(x["extra_info"]["question"])) + len(tokenizer.encode(x["extra_info"]["answer"])) <= 4000),
    })
    
    # Push the dataset to Hugging Face Hub
    dataset_dict.push_to_hub(args.ds_name)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nScript interrupted by user. Exiting gracefully...")
    except Exception as e:
        print(f"Error running script: {str(e)}")