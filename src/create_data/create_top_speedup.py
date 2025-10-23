import pandas as pd 
import os 
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import subprocess
import json
import numpy as np
import tempfile
import argparse
import threading
import signal
import psutil
import time



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


def run_with_timeout(cmd, shell=True, timeout=30, capture_output=True):
    """Run a command with timeout and proper cleanup of subprocess on failure."""
    process = None
    try:
        process = subprocess.Popen(
            cmd, 
            shell=shell, 
            stdout=subprocess.PIPE if capture_output else None,
            stderr=subprocess.PIPE if capture_output else None,
            text=True,
            preexec_fn=os.setsid
        )
        
        stdout, stderr = process.communicate(timeout=timeout)
        
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


def process_submission_with_speedup(row, temp_dir, results_list, lock, threshold=2.0, timeout=30):
    """
    Processes a candidate submission:
      - Compiles with O0 and O3 optimization levels
      - Runs benchmarks for both
      - Calculates speedup
      - If speedup >= threshold, stores results
    """
    pid = row['problem_id']
    sub_id = row['submission_id']
    
    # Get input file path (we already verified it exists and is not empty)
    input_file = os.path.join('test_cases', f"{pid}/input.0.txt")
    if not os.path.exists(input_file):
        print(f"Input file {input_file} does not exist")
        return None
    
    src_file = os.path.join('Project_CodeNet', 'data', str(pid), 'C', f'{sub_id}.c')
    try:
        with open(src_file, 'r') as f:
            code = f.read()
    except Exception as e:
        print(f"Error reading source file {src_file}: {e}")
        return None
    
    # Paths for O0 and O3 binaries
    binary_o0 = os.path.join(temp_dir, f"{pid}_{sub_id}_O0.bin")
    binary_o3 = os.path.join(temp_dir, f"{pid}_{sub_id}_O3.bin")
    
    try:
        # Compile with O0
        compile_cmd_o0 = f"gcc {src_file} -o {binary_o0} -O0 -lm"
        compile_result_o0 = run_with_timeout(compile_cmd_o0, timeout=30)
        
        if compile_result_o0['returncode'] != 0:
            print(f"Compilation with O0 failed for {pid}_{sub_id}: {compile_result_o0['stderr']}")
            return None
        
        # Compile with O3
        compile_cmd_o3 = f"gcc {src_file} -o {binary_o3} -O3 -lm"
        compile_result_o3 = run_with_timeout(compile_cmd_o3, timeout=30)
        
        if compile_result_o3['returncode'] != 0:
            print(f"Compilation with O3 failed for {pid}_{sub_id}: {compile_result_o3['stderr']}")
            return None
        
        # Generate assembly code for O0
        asm_file_o0 = os.path.join(temp_dir, f"{pid}_{sub_id}_O0.s")
        asm_cmd_o0 = f"gcc -S {src_file} -o {asm_file_o0} -O0 -lm"
        asm_result_o0 = run_with_timeout(asm_cmd_o0, timeout=30)
        
        if asm_result_o0['returncode'] != 0:
            print(f"Assembly generation with O0 failed for {pid}_{sub_id}: {asm_result_o0['stderr']}")
            return None
            
        # Generate assembly code for O3
        asm_file_o3 = os.path.join(temp_dir, f"{pid}_{sub_id}_O3.s")
        asm_cmd_o3 = f"gcc -S {src_file} -o {asm_file_o3} -O3 -lm"
        asm_result_o3 = run_with_timeout(asm_cmd_o3, timeout=30)
        
        if asm_result_o3['returncode'] != 0:
            print(f"Assembly generation with O3 failed for {pid}_{sub_id}: {asm_result_o3['stderr']}")
            return None
        
        # Read assembly code files
        with open(asm_file_o0, 'r') as f:
            o0_assembly_code_str = f.read()
        with open(asm_file_o3, 'r') as f:
            o3_assembly_code_str = f.read()
        
        question = f'Given the following C code and unoptimized assembly code, your task is to generate highly optimized x86-64 assembly code.\nC Code:\n\n```c\n{code}```\n\nUnoptimized Assembly Code:\n\n```assembly\n{o0_assembly_code_str}```\n\nOnly output the optimized assembly code. Do not include any other text. Do not write any comments in the assembly code. Wrap the assembly code in ```assembly``` tags.\nOptimized Assembly Code:\n'
        answer = '```assembly\n' + o3_assembly_code_str + '```'
        
        # Run benchmarks
        o0_result = run_hyperfine_benchmark(binary_o0, input_file, temp_dir, timeout)
        o3_result = run_hyperfine_benchmark(binary_o3, input_file, temp_dir, timeout)
        
        # Calculate speedup if both benchmarks succeeded
        if o0_result and o3_result and o3_result['mean'] > 0:
            speedup = o0_result['mean'] / o3_result['mean']
            
            # Only add submissions with speedup >= threshold
            if speedup >= threshold:
                with lock:
                    results_list.append({
                        'problem_id': pid,
                        'sub_id': sub_id,
                        'speedup': speedup,
                        'code': code,
                        'o0_time': o0_result['mean'],
                        'o3_time': o3_result['mean']
                    })
            
            return {
                'problem_id': pid,
                'submission_id': sub_id,
                'speedup': speedup,
                'o0_time': o0_result['mean'],
                'o3_time': o3_result['mean']
            }
        
        return None
    except Exception as e:
        print(f"Error processing submission {pid}_{sub_id}: {str(e)}")
        return None
    finally:
        # Always clean up binaries
        for binary in [binary_o0, binary_o3]:
            if os.path.exists(binary):
                try:
                    os.remove(binary)
                except Exception:
                    pass


def setup_signal_handlers(executor):
    """Set up signal handlers for graceful shutdown."""
    def signal_handler(sig, frame):
        print('Received signal to terminate. Shutting down gracefully...')
        executor.shutdown(wait=False, cancel_futures=True)
        os._exit(1)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


def find_top_speedup_submissions(top_n=10, num_submissions=10, threshold=2.0):
    """Finds the top N submissions (across all problems) with O0 to O3 speedup >= threshold."""
    # Create temp directory for benchmarks
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create the destination directory for code files
        output_path = os.path.join('C', 'TopCode')
        Path(output_path).mkdir(exist_ok=True, parents=True)
        
        # Gather all metadata CSV files (skipping problem_list.csv)
        metadata_path = os.path.join('Project_CodeNet', 'metadata')
        mpaths = [fl for fl in os.listdir(metadata_path)
                  if fl.endswith('.csv') and fl != 'problem_list.csv']
        # Sort files by an integer extracted from their name
        sorted_files = sorted(mpaths, key=lambda x: int(x.split('.')[0].split('p')[1]))
        
        # List to store all submissions with speedup >= threshold
        results_list = []
        lock = threading.Lock()
        
        # Use ThreadPoolExecutor to process submissions concurrently
        with ThreadPoolExecutor(max_workers=32) as executor:
            # Set up signal handlers for graceful shutdown
            setup_signal_handlers(executor)
            
            futures = []
            for metadata in sorted_files:
                csv_file = os.path.join(metadata_path, metadata)
                try:
                    df = pd.read_csv(csv_file)
                except Exception as e:
                    print(f"Error reading {csv_file}: {e}")
                    continue
                
                # Filter for accepted C submissions for valid problems
                df = df.loc[(df['language'] == 'C') & 
                            (df['status'] == 'Accepted')]
                
                # Submit each candidate submission from this metadata file as a separate task
                i = 0
                for _, row in df.iterrows():
                    futures.append(executor.submit(
                        process_submission_with_speedup, 
                        row, 
                        temp_dir, 
                        results_list, 
                        lock,
                        threshold, 
                    ))
                    i += 1 
                    if i == num_submissions:
                        break
            
            # Track progress with tqdm
            completed = 0
            total = len(futures)
            with tqdm(total=total, desc="Processing submissions") as pbar:
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        print(f"Error processing submission: {str(e)}")
                    finally:
                        completed += 1
                        pbar.update(1)
        
        # Sort all submissions by speedup and get top N
        top_submissions = sorted(results_list, key=lambda x: x['speedup'], reverse=True)
        if top_n > 0 and len(top_submissions) > top_n:
            top_submissions = top_submissions[:top_n]
        
        print(f"Found {len(top_submissions)} submissions with speedup >= {threshold}")
        
        # Write results to files with the pattern {pid}_{submissionidx}.c
        for idx, submission in enumerate(top_submissions):
            pid = submission['problem_id']
            sub_id = submission['sub_id']
            dest_file = os.path.join(output_path, f"{pid}_{sub_id}.c")
            try:
                with open(dest_file, 'w') as f:
                    f.write(submission['code'])
                print(f"Submission {idx+1}: Problem {pid}, Submission {sub_id}, Speedup = {submission['speedup']:.2f}x")
            except Exception as e:
                print(f"Error writing destination file {dest_file}: {e}")
        
        # Create a report file with summary statistics
        report_file = os.path.join('C', 'speedup_report.csv')
        report_data = []
        for submission in top_submissions:
            report_data.append({
                'problem_id': submission['problem_id'],
                'submission_id': submission['sub_id'],
                'speedup': submission['speedup'],
                'o0_time_ms': submission['o0_time'],
                'o3_time_ms': submission['o3_time']
            })
        
        if report_data:
            pd.DataFrame(report_data).to_csv(report_file, index=False)
            print(f"Report saved to {report_file}")
            # Mean overall speedup
            overall_speedup = np.mean([res['speedup'] for res in report_data])
            print(f"Overall mean speedup: {overall_speedup:.2f}x")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Find top N submissions with largest O0 to O3 speedup >= threshold')
    parser.add_argument('--top_n', type=int, default=5000, help='Number of top submissions to select')
    parser.add_argument('--num_submissions', type=int, default=10, help='Number of submissions to process per problem')
    parser.add_argument('--threshold', type=float, default=3.0, help='Minimum speedup threshold to consider (default: 2.0)')
    args = parser.parse_args()
    
    try:
        find_top_speedup_submissions(args.top_n, args.num_submissions, args.threshold)
    except KeyboardInterrupt:
        print("\nScript interrupted by user. Exiting gracefully...")
    except Exception as e:
        print(f"Error running script: {str(e)}")