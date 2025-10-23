#!/usr/bin/env python3
import os
import argparse
import subprocess
from subprocess import Popen, PIPE
from pathlib import Path
import concurrent.futures
from tqdm import tqdm
import shutil
from pathlib import Path
import psutil
import signal


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

def process_file(c_file, code_dir, test_dir, output_dir):
    """
    Process a single C file:
    1. Extract problem ID from filename
    2. Compile the code
    3. Run it with the corresponding input
    4. Store input and generated output in output_dir
    """
    try:
        # Extract problem ID (without the .c extension)
        cid = c_file.split('.')[0]
        pid, sub_id = cid.split('_')
        
        # Paths to source and input files
        src_path = os.path.join(code_dir, c_file)
        
        test_files = sorted([x for x in os.listdir(f'{test_dir}/{pid}') if 'input' in x], key = lambda x: int(x.split('.')[1]))[:10]
        if not test_files:
            print(f"No test files found for {pid}. Skipping.")
            return {"pid": pid, "status": "skipped", "reason": "no test files"}
        

        # Create output directory if it doesn't exist
        Path(f'{output_dir}/{pid}_{sub_id}').mkdir(exist_ok=True, parents=True)
        
        for test_file in test_files:
            # Compile the C file
            binary_path = os.path.join(os.getcwd(), f"{pid}_{sub_id}.bin")
            compile_cmd = f"gcc {src_path} -o {binary_path} -lm"
            
            # Use our robust function for compilation
            compile_result = run_with_timeout(compile_cmd, timeout=30)
            
            if compile_result['returncode'] != 0:
                if os.path.exists(binary_path):
                    os.remove(binary_path)
                return {"pid": pid, "status": "failed", "reason": "compilation error"}
            
            with open(os.path.join(test_dir, pid, test_file), 'r') as f:
                input_content = f.read()
            
            # Run the compiled binary with the input using our robust function
            run_cmd = f"{binary_path}"
            run_result = run_with_timeout(
                run_cmd, 
                shell=False, 
                timeout=60, 
                input_data=input_content.encode()
            )
            
            # Clean up binary
            if os.path.exists(binary_path):
                os.remove(binary_path)
            
            # Check for runtime errors or timeout
            if run_result['returncode'] != 0:
                error_msg = run_result['stderr']
                if error_msg and isinstance(error_msg, bytes):
                    error_msg = error_msg.decode()
                reason = "timeout" if "timed out" in str(error_msg) else "runtime error"
                return {"pid": pid, "status": "failed", "reason": reason}
            
            # Get output content
            output_content = run_result['stdout']
            if isinstance(output_content, bytes):
                output_content = output_content.decode()
            
            # Write input and output to files in output_dir
            with open(os.path.join(output_dir, f"{pid}_{sub_id}", test_file), 'w') as f:
                f.write(input_content)
            
            with open(os.path.join(output_dir, f"{pid}_{sub_id}", test_file.replace('input', 'output')), 'w') as f:
                f.write(output_content)
        
        return {"cid": cid, "status": "success"}
    
    except Exception as e:
        return {"cid": cid, "status": "failed", "reason": str(e)}

def main():
    parser = argparse.ArgumentParser(description='Extract inputs and outputs for top speedup problems')
    parser.add_argument('--code_dir', default='C/TopCode', help='Directory containing top code files')
    parser.add_argument('--test_dir', default='test_cases', help='Directory containing test cases')
    parser.add_argument('--output_dir', default='C/TopIO', help='Directory to store inputs and outputs')
    parser.add_argument('--max_workers', type=int, default=64, help='Maximum number of parallel workers')
    parser.add_argument('--skip_ids', type=str, default='')
    parser.add_argument('--failed_log', default='failed_and_skipped_cids.txt', help='File to log failed and skipped CIDs')
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    skip_ids = args.skip_ids.split(',')
    
    # Get list of C files in code_dir
    c_files = [f for f in os.listdir(args.code_dir) if f.endswith('.c') and f.split('.c')[0] not in skip_ids]
    
    if not c_files:
        print(f"No C files found in {args.code_dir}")
        return
    
    print(f"Processing {len(c_files)} files...")
    
    # Process files in parallel
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(
                process_file, 
                c_file, 
                args.code_dir, 
                args.test_dir, 
                args.output_dir
            ): c_file for c_file in c_files
        }
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(c_files), desc="Extracting I/O"):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                c_file = futures[future]
                print(f"Error processing {c_file}: {e}")
                # Add unhandled exceptions to results
                cid = c_file.split('.')[0]
                results.append({"cid": cid, "status": "failed", "reason": str(e)})
    
    # Summarize results
    successful = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] == "failed"]
    skipped = [r for r in results if r["status"] == "skipped"]
    
    print(f"\nSummary:")
    print(f"  Total files: {len(c_files)}")
    print(f"  Successfully processed: {len(successful)}")
    print(f"  Failed: {len(failed)}")
    print(f"  Skipped: {len(skipped)}")
    
    if failed:
        print("\nFailed files:")
        for f in failed:
            print(f"  {f['cid'] if 'cid' in f else f['pid']}: {f['reason']}")
    
    if skipped:
        print("\nSkipped files:")
        for s in skipped:
            print(f"  {s['cid'] if 'cid' in s else s['pid']}: {s['reason']}")
    
    # Save failed and skipped CIDs to file
    if failed or skipped:
        with open(args.failed_log, 'w') as f:
            f.write("# Failed and skipped CIDs from processing\n")
            f.write("# Format: CID,STATUS,REASON\n\n")
            
            # Write failed entries
            for entry in failed:
                cid = entry.get('cid', entry.get('pid', 'unknown'))
                reason = entry.get('reason', 'unknown')
                f.write(f"{cid},failed,\"{reason}\"\n")
            
            # Write skipped entries
            for entry in skipped:
                cid = entry.get('cid', entry.get('pid', 'unknown'))
                reason = entry.get('reason', 'unknown')
                f.write(f"{cid},skipped,\"{reason}\"\n")
        
        print(f"\nFailed and skipped CIDs saved to {args.failed_log}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nScript interrupted by user. Exiting gracefully...")
    except Exception as e:
        print(f"Error running script: {str(e)}") 