import os
import tempfile
import subprocess
import psutil
import json
import numpy as np
from subprocess import Popen, PIPE, TimeoutExpired, CalledProcessError
import torch.distributed as dist


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


def run_hyperfine_benchmark(binary_path, input_file, temp_dir, timeout=30):
    """Run hyperfine benchmark on a binary with given input file."""
    try:
        # First verify the program doesn't hang with a quick test run
        test_cmd = f"{binary_path} < {input_file}"
        test_result = run_with_timeout(test_cmd, timeout=10)
        
        if test_result['returncode'] != 0:
            return None
        
        # If it completes successfully, run the benchmark
        temp_output = os.path.join(temp_dir, f"{os.path.basename(binary_path)}_hyperfine.json")
        benchmark_cmd = (
            f"hyperfine --warmup 3 --runs 10 --input {input_file} --export-json {temp_output} "
            f"--time-unit millisecond '{binary_path}'"
        )
        
        benchmark_result = run_with_timeout(benchmark_cmd, timeout=timeout)
        
        if benchmark_result['returncode'] != 0:
            return None
        
        # Check if the file exists before trying to read it
        if not os.path.exists(temp_output):
            return None
            
        # Read benchmark results
        try:
            with open(temp_output, 'r') as f:
                benchmark_data = json.load(f)
            
            # Check if there are any results
            if 'results' not in benchmark_data or not benchmark_data['results']:
                return None
                
            # Extract median time - fix for StopIteration by making sure we handle empty results
            found_result = False
            for res in benchmark_data['results']:
                if res['command'] == f"{binary_path}":
                    return {
                        'mean': res['mean'],
                        'median': res['median']
                    }
                
            # No results found
            return None
            
        except (json.JSONDecodeError, Exception) as e:
            return None
                
        return None
    except Exception as e:
        return None


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """
    Compute the speedup of the solution assembly code over the unoptimized assembly code.

    Arguments:
      solution_str (str): the full assembly source.
      ground_truth (str): the expected assembly code (not used for evaluation).
      extra_info (dict):
         {
           "inputs":  [str, ...],   # each test's stdin
           "outputs": [str, ...],   # corresponding expected stdout
           "unoptimized_assembly": str,  # the unoptimized assembly code to compare against
           "unoptimized_compiled": bytes  # the precompiled unoptimized assembly binary
         }

    Returns:
      float reward:
        1 * (speedup) if solution is correct and faster than unoptimized assembly
        0 if solution is incorrect or unoptimized assembly is faster
    """
    q_val = 10
    
    # Clean up solution string
    solution_str = solution_str.replace("```assembly\n", "").replace("```", "")
    if solution_str.strip() == extra_info["unoptimized_assembly"].replace("```assembly\n", "").replace("```", "").strip():
        return 1
    
    # First check if the solution is correct and get the compiled binary
    solution_correctness, solution_binary = check_correctness(solution_str, ground_truth, extra_info)
    
    if solution_correctness < 1.0:  # Not all test cases passed
        return solution_correctness
        #return 0
    if solution_binary is None:
        # If we didn't get a binary for some reason, we can't proceed
        return solution_correctness
        #return 0
    inputs = extra_info["inputs"][:10]
    
    # Check if we have a precompiled unoptimized binary
    precompiled_unoptimized = extra_info['unoptimized_compiled']
    
    # Benchmark the solution vs unoptimized assembly
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Write the already compiled solution binary to file
            solution_bin_file = os.path.join(temp_dir, "solution.bin")
            with open(solution_bin_file, 'wb') as f:
                f.write(solution_binary)
            # Make the binary executable
            os.chmod(solution_bin_file, 0o755)
            
            # Handle unoptimized binary
            unopt_bin_file = os.path.join(temp_dir, "unopt.bin")
            
            with open(unopt_bin_file, 'wb') as f:
                f.write(precompiled_unoptimized)
            # Make the binary executable
            os.chmod(unopt_bin_file, 0o755)

            
            # Measure speedup for each input
            speedups = []
            for i, input_text in enumerate(inputs):
                # Create input file
                input_file = os.path.join(temp_dir, f"input_{i}.txt")
                with open(input_file, 'w') as f:
                    f.write(input_text)
                
                # Run benchmarks for this input
                unopt_result = run_hyperfine_benchmark(unopt_bin_file, input_file, temp_dir, timeout=60)
                solution_result = run_hyperfine_benchmark(solution_bin_file, input_file, temp_dir, timeout=60)
                
                # Calculate speedup if both benchmarks succeeded
                if unopt_result and solution_result and solution_result['mean'] > 0:
                    speedup = unopt_result['mean'] / solution_result['mean']
                    speedups.append(speedup)
            
            # Calculate average speedup across all inputs
            if speedups:
                avg_speedup = np.mean(speedups)
                # Only return positive reward if solution is faster than unoptimized assembly
                if avg_speedup > 1.0:
                    return 1 + q_val * avg_speedup
                    #return avg_speedup
            
            return 1
            return 0
    
    except Exception as e:
        return -1  # Any unhandled exception results in no reward
        #return 0

def check_correctness(solution_str, ground_truth, extra_info=None):
    """
    Compile & test a piece of assembly code against multiple I/O pairs.

    Arguments:
      solution_str (str): the full assembly source.
      ground_truth (str): the expected assembly code (not used for evaluation).
      extra_info (dict):
         {
           "inputs":  [str, ...],   # each test's stdin
           "outputs": [str, ...],   # corresponding expected stdout
           "unoptimized_assembly": str,  # the unoptimized assembly code
           "unoptimized_compiled": bytes (optional)  # precompiled unoptimized binary
         }

    Returns:
      (float, bytes) tuple of (reward, compiled_binary):
        reward:
          -1   compilation/link failure
          -0.5   runtime error on any test
          ≥0   (#passed tests) / (total tests)
        compiled_binary:
          The compiled binary data if successful, or None if compilation failed
    """
    
    if not extra_info or "inputs" not in extra_info or "outputs" not in extra_info or "unoptimized_assembly" not in extra_info:
        return (-1, None)

    
    inputs = extra_info["inputs"][:10]
    outputs = extra_info["outputs"][:10]
    
    if len(inputs) == 0 or len(inputs) != len(outputs):
        return (-1, None)

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Write assembly to file
            asm_file = os.path.join(temp_dir, "test.s")
            with open(asm_file, 'w') as f:
                f.write(solution_str)
            
            # Compile the assembly with -lm to match compilation in evaluate.py
            bin_file = os.path.join(temp_dir, "test.bin")
            compile_cmd = f"gcc {asm_file} -o {bin_file} -lm"
            
            compile_result = run_with_timeout(compile_cmd, timeout=30)
            if compile_result['returncode'] != 0:
                return (-1, None)  # Compilation failure
            
            passed_tests = 0
            total_tests = len(inputs)
            
            # Test each input
            for i, (input_text, expected_output) in enumerate(zip(inputs, outputs)):
                # Run the binary with input
                run_cmd = f"{bin_file}"
                run_result = run_with_timeout(
                    run_cmd,
                    shell=False,
                    timeout=120,  # Match the 120 second timeout from evaluate.py
                    input_data=input_text.encode()
                )
                
                if run_result['returncode'] != 0:
                    return (-0.5, None)  # Runtime error on any test
                
                # Get actual output
                actual_output = run_result['stdout']
                if isinstance(actual_output, bytes):
                    actual_output = actual_output.decode()
                
                # Use raw comparison to match exact behavior
                if actual_output == expected_output:
                    passed_tests += 1
            
            result = passed_tests / total_tests
            
            # Read the binary data from the compiled file
            try:
                with open(bin_file, 'rb') as f:
                    binary_data = f.read()
                return (result, binary_data)
            except Exception:
                return (result, None)
    
    except Exception as e: 
        return (-1, None)




    