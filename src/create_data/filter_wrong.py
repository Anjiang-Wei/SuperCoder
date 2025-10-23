import requests
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
            stdin=subprocess.PIPE if input_data else None,
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


def test_assembly_code(assembly_code, inputs, outputs):
    """
    Test the generated assembly code against inputs and check if outputs match.
    
    Args:
        assembly_code: The assembly code to test
        inputs: List of input strings
        outputs: List of expected output strings
        
    Returns:
        tuple: (is_correct, error_message)
    """
    assembly_code = assembly_code.replace('```assembly', '').replace('```', '')
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Write assembly to file
            asm_file = os.path.join(temp_dir, "test.s")
            with open(asm_file, 'w') as f:
                f.write(assembly_code)
            
            # Compile the assembly with -lm to match idk.py compilation
            bin_file = os.path.join(temp_dir, "test.bin")
            compile_cmd = f"gcc {asm_file} -o {bin_file} -lm"
            
            compile_result = run_with_timeout(compile_cmd, timeout=30)
            if compile_result['returncode'] != 0:
                return False, f"Compilation error: {compile_result['stderr']}"
            
            # Test each input
            for i, (input_text, expected_output) in enumerate(zip(inputs, outputs)):
                # Run the binary with input using the same timeout (120) as in idk.py
                run_cmd = f"{bin_file}"
                run_result = run_with_timeout(
                    run_cmd,
                    shell=False,
                    timeout=120,  # Match the 120 second timeout from idk.py
                    input_data=input_text.encode()
                )
                
                if run_result['returncode'] != 0:
                    return False, f"Runtime error on test {i+1}: {run_result['stderr']}"
                
                # Get actual output
                actual_output = run_result['stdout']
                if isinstance(actual_output, bytes):
                    actual_output = actual_output.decode()
                
                # Use raw comparison instead of stripping to match exact behavior
                if actual_output != expected_output:
                    return False, f"Output mismatch on test {i+1}\nExpected: {expected_output}\nActual: {actual_output}"
            
            # All tests passed
            return True, ""
    
    except Exception as e:
        return False, f"Error testing assembly: {str(e)}"


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
            f"hyperfine --warmup 1 --runs 1 --input {input_file} --export-json {temp_output} "
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


def benchmark_assembly_code(unoptimized_assembly, optimized_assembly, inputs):
    """
    Benchmark the unoptimized and optimized assembly code to measure speedup.
    
    Args:
        unoptimized_assembly: The unoptimized assembly code
        optimized_assembly: The optimized assembly code to benchmark
        inputs: List of input strings for testing
        
    Returns:
        dict: Benchmark results including speedup or None if benchmarking failed
    """
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Write unoptimized assembly to file
            unopt_asm_file = os.path.join(temp_dir, "unopt.s")
            with open(unopt_asm_file, 'w') as f:
                f.write(unoptimized_assembly)
            
            # Write optimized assembly to file
            opt_asm_file = os.path.join(temp_dir, "opt.s")
            with open(opt_asm_file, 'w') as f:
                f.write(optimized_assembly)
            
            # Compile both assembly files
            unopt_bin_file = os.path.join(temp_dir, "unopt.bin")
            opt_bin_file = os.path.join(temp_dir, "opt.bin")
            
            compile_cmd_unopt = f"gcc {unopt_asm_file} -o {unopt_bin_file} -lm"
            compile_result_unopt = run_with_timeout(compile_cmd_unopt, timeout=30)
            
            compile_cmd_opt = f"gcc {opt_asm_file} -o {opt_bin_file} -lm"
            compile_result_opt = run_with_timeout(compile_cmd_opt, timeout=30)
            
            if compile_result_unopt['returncode'] != 0 or compile_result_opt['returncode'] != 0:
                print("Compilation error during benchmarking")
                return None
            
            # For each input, create a temporary input file and benchmark
            speedups = []
            for i, input_text in enumerate(inputs):
                input_file = os.path.join(temp_dir, f"input_{i}.txt")
                with open(input_file, 'w') as f:
                    f.write(input_text)
                
                # Run benchmarks for this input
                unopt_result = run_hyperfine_benchmark(unopt_bin_file, input_file, temp_dir, timeout=60)
                opt_result = run_hyperfine_benchmark(opt_bin_file, input_file, temp_dir, timeout=60)
                
                # Calculate speedup if both benchmarks succeeded
                if unopt_result and opt_result and opt_result['mean'] > 0:
                    speedup = unopt_result['mean'] / opt_result['mean']
                    speedups.append(speedup)
            
            # Calculate average speedup across all inputs
            if speedups:
                avg_speedup = np.mean(speedups)
                return {
                    'speedup': avg_speedup,
                    'unopt_time': unopt_result['mean'] if 'unopt_result' in locals() else None,
                    'opt_time': opt_result['mean'] if 'opt_result' in locals() else None,
                    'num_test_cases': len(speedups)
                }
            
            return None
    except Exception as e:
        print(f"Error benchmarking assembly: {str(e)}")
        return None


def process_single_test_case(problem_idx, test_case_idx, item, gen_res, input_text, expected_output):
    """
    Process a single test case for a problem.
    
    Args:
        problem_idx: Index of the problem
        test_case_idx: Index of the test case within the problem
        item: The problem item data
        gen_res: Generated optimized assembly code
        input_text: The input for this test case
        expected_output: The expected output for this test case
        
    Returns:
        dict: Result information for this specific test case
    """
    try:
        unoptimized_assembly = item['extra_info']['unoptimized_assembly']
        
        # Normalize assembly code
        gen_res = gen_res.replace('```assembly', '').replace('```', '')
        unoptimized_assembly = unoptimized_assembly.replace('```assembly', '').replace('```', '')
        
        # Test just this single test case
        is_correct = False
        error_msg = ""
        
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                # Write assembly to file
                asm_file = os.path.join(temp_dir, "test.s")
                with open(asm_file, 'w') as f:
                    f.write(gen_res)
                
                # Compile the assembly
                bin_file = os.path.join(temp_dir, "test.bin")
                compile_cmd = f"gcc {asm_file} -o {bin_file} -lm"
                
                compile_result = run_with_timeout(compile_cmd, timeout=30)
                if compile_result['returncode'] != 0:
                    error_msg = f"Compilation error: {compile_result['stderr']}"
                else:
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
            'speedup': 0.0,  # Default to 0 speedup for failed cases
            'unopt_time': None,
            'opt_time': None
        }
        
        # Only benchmark if the test case passed
        if is_correct:
            try:
                with tempfile.TemporaryDirectory() as temp_dir:
                    # Write unoptimized assembly to file
                    unopt_asm_file = os.path.join(temp_dir, "unopt.s")
                    with open(unopt_asm_file, 'w') as f:
                        f.write(unoptimized_assembly)
                    
                    # Write optimized assembly to file
                    opt_asm_file = os.path.join(temp_dir, "opt.s")
                    with open(opt_asm_file, 'w') as f:
                        f.write(gen_res)
                    
                    # Compile both assembly files
                    unopt_bin_file = os.path.join(temp_dir, "unopt.bin")
                    opt_bin_file = os.path.join(temp_dir, "opt.bin")
                    
                    compile_cmd_unopt = f"gcc {unopt_asm_file} -o {unopt_bin_file} -lm"
                    compile_result_unopt = run_with_timeout(compile_cmd_unopt, timeout=30)
                    
                    compile_cmd_opt = f"gcc {opt_asm_file} -o {opt_bin_file} -lm"
                    compile_result_opt = run_with_timeout(compile_cmd_opt, timeout=30)
                    
                    if compile_result_unopt['returncode'] == 0 and compile_result_opt['returncode'] == 0:
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
                            result.update({
                                'speedup': speedup,
                                'unopt_time': unopt_result['mean'],
                                'opt_time': opt_result['mean']
                            })
            except Exception as e:
                print(f"Error benchmarking test case {test_case_idx} for problem {problem_idx}: {str(e)}")
        
        return result
    except Exception as e:
        return {
            'problem_idx': problem_idx,
            'test_case_idx': test_case_idx,
            'status': 'failed',
            'error_msg': f"Exception: {str(e)}",
            'speedup': 0.0,
            'unopt_time': None,
            'opt_time': None
        }


def main(ds_name):
    ds_dct = {}
    for split in load_dataset(ds_name).keys():
        test_dataset = load_dataset(ds_name, split=split)
        test_dataset = test_dataset.map(lambda example, idx: {"idx": idx}, with_indices=True)
        # take a look at the first 10 items
        # test_dataset = test_dataset.select(range(10))
        
        
        gen_results = [item['extra_info']['answer'] for item in test_dataset]
        model_name = 'gt'

        # Test the generated assembly code against inputs and outputs using multithreading
        print("Testing and benchmarking assembly code...")
        
        # Set up variables to track progress and results
        max_workers = num_workers  # Use the same number of workers as specified for the inference
        test_case_results = []
        tasks = []
        
        # Create tasks for each (problem, test_case) pair
        for problem_idx, (item, gen_res) in enumerate(zip(test_dataset, gen_results)):
            inputs = item['extra_info']['inputs'][:10]
            outputs = item['extra_info']['outputs'][:10]
            
            for test_case_idx, (input_text, expected_output) in enumerate(zip(inputs, outputs)):
                tasks.append((problem_idx, test_case_idx, item, gen_res, input_text, expected_output))
        
        total_tasks = len(tasks)
        print(f"Total tasks to process: {total_tasks}")
        
        # Use ThreadPoolExecutor for parallel processing of all (problem, test_case) pairs
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks to the thread pool
            futures = {
                executor.submit(process_single_test_case, *task): task
                for task in tasks
            }
            
            # Process results as they complete
            for future in tqdm(concurrent.futures.as_completed(futures), total=total_tasks, desc="Processing test cases"):
                try:
                    result = future.result()
                    test_case_results.append(result)
                except Exception as e:
                    task = futures[future]
                    problem_idx, test_case_idx = task[0], task[1]
                    test_case_results.append({
                        'problem_idx': problem_idx,
                        'test_case_idx': test_case_idx,
                        'status': 'failed',
                        'error_msg': f"Unhandled exception: {str(e)}",
                        'speedup': 0.0,
                        'unopt_time': None,
                        'opt_time': None
                    })
        
        # Organize results by problem - initialize with problem data first
        problem_results = {}
        for problem_idx, (item, gen_res) in enumerate(zip(test_dataset, gen_results)):
            # Get the actual index from the dataset to ensure consistency
            idx = item['idx'] if 'idx' in item else problem_idx
            
            # Store the generated assembly code for each problem
            problem_results[idx] = {
                'correct': True,  # Will be set to False if any test fails
                'speedup': 0.0,
                'test_cases': [],
                'num_test_cases': 0,
                'generated_assembly': gen_res,  # Save the generated assembly
                'unoptimized_assembly': item['extra_info']['unoptimized_assembly']  # Also save the unoptimized assembly for reference
            }
        
        # Process test case results
        for result in test_case_results:
            problem_idx = result['problem_idx']
            
            # Ensure the problem exists in the results dictionary
            if problem_idx not in problem_results:
                print(f"Warning: Test case refers to missing problem {problem_idx}")
                continue
            
            # Add this test case result
            problem_results[problem_idx]['test_cases'].append({
                'test_case_idx': result['test_case_idx'],
                'status': result['status'],
                'error_msg': result['error_msg'],
                'speedup': result['speedup'],
                'unopt_time': result['unopt_time'],
                'opt_time': result['opt_time']
            })
            
            # Update problem correctness
            if result['status'] != 'success':
                problem_results[problem_idx]['correct'] = False
        
        # Calculate speedups for problems (average of successful test cases)
        for problem_idx, problem in problem_results.items():
            successful_test_cases = [tc for tc in problem['test_cases'] if tc['status'] == 'success' and tc['speedup'] > 0]
            problem['num_test_cases'] = len(problem['test_cases'])
            
            if successful_test_cases:
                problem['speedup'] = np.mean([tc['speedup'] for tc in successful_test_cases])
        
        
        failed_problem_idxs = [problem_idx for problem_idx, problem in problem_results.items() if not problem['correct']]
        ds_dct[split] = test_dataset.filter(lambda x: x['idx'] not in failed_problem_idxs)
    
    ds = DatasetDict(ds_dct)
    ds.push_to_hub(f'{ds_name}_filtered')


if __name__ == "__main__":
    fire.Fire(main)