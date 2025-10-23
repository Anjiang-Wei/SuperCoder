import requests
from transformers import AutoTokenizer
import os
import ray
from datasets import load_dataset
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







def run_hyperfine_benchmark(binary_path, input_file, temp_dir, timeout=30):
    """Run hyperfine benchmark on a binary with given input file."""
    try:
        # First verify the program doesn't hang with a quick test run
        test_cmd = f"{binary_path} < {input_file}"
        test_result = run_with_timeout(test_cmd, timeout=10)
        
        if test_result['returncode'] != 0:
            print(f"Test run failed: {test_result['stderr']}")
            return None
        
        # If it completes successfully, run the benchmark
        temp_output = os.path.join(temp_dir, f"{os.path.basename(binary_path)}_hyperfine.json")
        benchmark_cmd = (
            f"hyperfine --warmup 3 --runs 10 --input {input_file} --export-json {temp_output} "
            f"--time-unit millisecond '{binary_path}'"
        )
        
        benchmark_result = run_with_timeout(benchmark_cmd, timeout=timeout)
        
        if benchmark_result['returncode'] != 0:
            print(f"Benchmark failed: {benchmark_result['stderr']}")
            return None
        
        # Check if the file exists before trying to read it
        if not os.path.exists(temp_output):
            print(f"Benchmark output file {temp_output} does not exist")
            return None
            
        # Read benchmark results
        try:
            with open(temp_output, 'r') as f:
                benchmark_data = json.load(f)
                
            # Extract median time
            for res in benchmark_data['results']:
                if res['command'] == f"{binary_path}":
                    return {
                        'mean': res['mean'],
                        'median': res['median']
                    }
        except json.JSONDecodeError as e:
            print(f"Error parsing benchmark results: {str(e)}")
        except Exception as e:
            print(f"Error reading benchmark results: {str(e)}")
                
        return None
    except Exception as e:
        print(f"Error during benchmark: {str(e)}")
        return None
    finally:
        # Clean up temp output file if exists
        if 'temp_output' in locals() and os.path.exists(temp_output):
            try:
                os.remove(temp_output)
            except Exception:
                pass





def process_single_test_case(problem_idx, test_case_idx, item, compiled_binary, input_text, expected_output):
    """
    Process a single test case for a problem using a pre-compiled binary.
    
    Args:
        problem_idx: Index of the problem
        test_case_idx: Index of the test case within the problem
        item: The problem item data
        compiled_binary: Pre-compiled binary of the generated assembly
        input_text: The input for this test case
        expected_output: The expected output for this test case
        
    Returns:
        dict: Result information for this specific test case
    """
    try:
        # Get precompiled unoptimized binary if available
        unoptimized_compiled = item['extra_info'].get('unoptimized_compiled', None)
        
        # Test just this single test case
        is_correct = False
        error_msg = ""
        
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                # Write the compiled binary to a file
                bin_file = os.path.join(temp_dir, "test.bin")
                with open(bin_file, 'wb') as f:
                    f.write(compiled_binary)
                
                # Make the binary executable
                os.chmod(bin_file, 0o755)
                
                # Run the binary with this specific input
                run_cmd = f"{bin_file}"
                run_result = run_with_timeout(
                    run_cmd,
                    shell=False,
                    timeout=120,
                    input_data=input_text.encode()
                )
                
                if run_result['returncode'] != 0:
                    error_msg = f"Runtime error: {run_result['stderr']}"
                else:
                    # Get actual output
                    actual_output = run_result['stdout']
                    if isinstance(actual_output, bytes):
                        actual_output = actual_output.decode()
                    
                    # Compare outputs
                    if actual_output == expected_output:
                        is_correct = True
                    else:
                        error_msg = f"Output mismatch\nExpected: {expected_output}\nActual: {actual_output}"
        except Exception as e:
            error_msg = f"Error testing assembly: {str(e)}"
        
        # Initialize result with default values
        result = {
            'problem_idx': problem_idx,
            'test_case_idx': test_case_idx,
            'status': 'success' if is_correct else 'failed',
            'error_msg': error_msg,
            'speedup': 1.0 if not is_correct else 0.0,  # Default to 1.0 speedup for failed cases
            'unopt_time': None,
            'opt_time': None
        }
        
        # Only benchmark if the test case passed
        if is_correct:
            try:
                with tempfile.TemporaryDirectory() as temp_dir:
                    # Write the optimized binary to a file
                    opt_bin_file = os.path.join(temp_dir, "opt.bin")
                    with open(opt_bin_file, 'wb') as f:
                        f.write(compiled_binary)
                    
                    # Make the binary executable
                    os.chmod(opt_bin_file, 0o755)
                    
                    # For unoptimized, use precompiled binary
                    unopt_bin_file = os.path.join(temp_dir, "unopt.bin")
                    
                    # Save precompiled binary to file
                    with open(unopt_bin_file, 'wb') as f:
                        f.write(unoptimized_compiled)
                    
                    # Make the binary executable
                    os.chmod(unopt_bin_file, 0o755)
                    
                    # Create input file for this test case
                    input_file = os.path.join(temp_dir, "input.txt")
                    with open(input_file, 'w') as f:
                        f.write(input_text)
                    
                    # Run benchmarks for this input
                    unopt_result = run_hyperfine_benchmark(unopt_bin_file, input_file, temp_dir, timeout=60)
                    opt_result = run_hyperfine_benchmark(opt_bin_file, input_file, temp_dir, timeout=60)
                    
                    # Calculate speedup if both benchmarks succeeded
                    if unopt_result and opt_result and opt_result['mean'] > 0:
                        speedup = unopt_result['mean'] / opt_result['mean']
                        # If generated code is slower, set speedup to 1.0
                        if speedup < 1.0:
                            speedup = 1.0
                        result.update({
                            'speedup': speedup,
                            'unopt_time': unopt_result['mean'],
                            'opt_time': opt_result['mean']
                        })
                    else:
                        # If benchmark fails, set speedup to 1.0
                        result['speedup'] = 1.0
            except Exception as e:
                print(f"Error benchmarking test case {test_case_idx} for problem {problem_idx}: {str(e)}")
                # If benchmarking fails, set speedup to 1.0
                result['speedup'] = 1.0
        
        return result
    except Exception as e:
        return {
            'problem_idx': problem_idx,
            'test_case_idx': test_case_idx,
            'status': 'failed',
            'error_msg': f"Exception: {str(e)}",
            'speedup': 1.0,  # Default to 1.0 speedup for failed cases
            'unopt_time': None,
            'opt_time': None
        }







def test_benchmark_generated_assembly(gen_results_by_problem, test_dataset, num_workers):
    # Pre-compile all assembly code generated by the model
    print("Pre-compiling generated assembly code...")
    compiled_binaries = {}
    compilation_failures = set()  # Track problems with compilation failures
    
    # Track if generated assembly is identical to unoptimized assembly
    identical_assembly_samples = {}
    
    for problem_idx, problem_samples in enumerate(gen_results_by_problem):
        problem_compilation_success = False  # Track if any sample for this problem compiles
        unoptimized_assembly = test_dataset[problem_idx]['extra_info']['unoptimized_assembly']
        
        for sample in problem_samples:
            sample_id = sample['sample_id']
            gen_res = sample['output']
            
            # Clean up assembly code
            gen_res = gen_res.replace('```assembly\n', '').replace('```', '')
            
            # Check if generated assembly is identical to unoptimized assembly
            if is_same_assembly(gen_res, unoptimized_assembly):
                identical_assembly_samples[(problem_idx, sample_id)] = True
                print(f"Generated assembly for problem {problem_idx}, sample {sample_id} is identical to unoptimized assembly")
            else:
                identical_assembly_samples[(problem_idx, sample_id)] = False
            
            # Compile the assembly
            success, result = compile_assembly(gen_res)
            
            if success:
                compiled_binaries[(problem_idx, sample_id)] = result
                problem_compilation_success = True
            else:
                print(f"Failed to compile assembly for problem {problem_idx}, sample {sample_id}: {result}")
                # Store None, so we can skip this sample later
                compiled_binaries[(problem_idx, sample_id)] = None
        
        # If none of the samples for this problem compiled successfully, mark it as a compilation failure
        if not problem_compilation_success:
            compilation_failures.add(problem_idx)

    # Test the generated assembly code against inputs and outputs using multithreading
    print("Testing and benchmarking assembly code...")
    
    # Set up variables to track progress and results
    max_workers = num_workers  # Use the same number of workers as specified for the inference
    test_case_results = []
    tasks = []
    
    # Create tasks for each (problem, sample_id, test_case) triplet
    for problem_idx, problem_samples in enumerate(gen_results_by_problem):
        item = test_dataset[problem_idx]
        inputs = item['extra_info']['inputs'][:10]
        outputs = item['extra_info']['outputs'][:10]
        
        for sample in problem_samples:
            sample_id = sample['sample_id']
            
            # Skip if compilation failed for this sample
            if compiled_binaries.get((problem_idx, sample_id)) is None:
                continue
            
            # If assembly is identical to unoptimized, skip testing and set speedup to 1x
            if identical_assembly_samples.get((problem_idx, sample_id), False):
                for test_case_idx in range(len(inputs)):
                    test_case_results.append({
                        'problem_idx': problem_idx,
                        'sample_id': sample_id,
                        'test_case_idx': test_case_idx,
                        'status': 'success',  # Assume success since it's identical to unoptimized
                        'error_msg': "",
                        'speedup': 1.0,  # Set speedup to 1x since it's identical
                        'unopt_time': None,
                        'opt_time': None,
                        'identical_to_unoptimized': True  # Mark as identical
                    })
                continue
            
            compiled_binary = compiled_binaries[(problem_idx, sample_id)]
            
            for test_case_idx, (input_text, expected_output) in enumerate(zip(inputs, outputs)):
                tasks.append((problem_idx, sample_id, test_case_idx, item, compiled_binary, input_text, expected_output))
    
    total_tasks = len(tasks)
    print(f"Total tasks to process: {total_tasks}")
    
    # Modified process_single_test_case function to include sample_id
    def process_single_test_case_with_sample(problem_idx, sample_id, test_case_idx, item, compiled_binary, input_text, expected_output):
        result = process_single_test_case(problem_idx, test_case_idx, item, compiled_binary, input_text, expected_output)
        result['sample_id'] = sample_id  # Add sample_id to the result
        result['identical_to_unoptimized'] = identical_assembly_samples.get((problem_idx, sample_id), False)  # NEW
        return result
    
    # Use ThreadPoolExecutor for parallel processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks to the thread pool
        futures = {
            executor.submit(process_single_test_case_with_sample, *task): task
            for task in tasks
        }
        
        # Process results as they complete
        for future in tqdm(concurrent.futures.as_completed(futures), total=total_tasks, desc="Processing test cases"):
            try:
                result = future.result()
                test_case_results.append(result)
            except Exception as e:
                task = futures[future]
                problem_idx, sample_id, test_case_idx = task[0], task[1], task[2]
                test_case_results.append({
                    'problem_idx': problem_idx,
                    'sample_id': sample_id,
                    'test_case_idx': test_case_idx,
                    'status': 'failed',
                    'error_msg': f"Unhandled exception: {str(e)}",
                    'speedup': 1.0,  # Set to 1.0 for failed cases - NEW
                    'unopt_time': None,
                    'opt_time': None,
                    'identical_to_unoptimized': identical_assembly_samples.get((problem_idx, sample_id), False)  # NEW
                })
    
    # Group results by problem_idx and sample_id
    results_by_problem_and_sample = {}
    for result in test_case_results:
        problem_idx = result['problem_idx']
        sample_id = result['sample_id']
        
        key = (problem_idx, sample_id)
        if key not in results_by_problem_and_sample:
            results_by_problem_and_sample[key] = []
        
        results_by_problem_and_sample[key].append(result)
    
    return test_case_results, results_by_problem_and_sample, compilation_failures, identical_assembly_samples