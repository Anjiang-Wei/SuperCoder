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


def compile_assembly(assembly_code):
    """
    Compile assembly code and return the compiled binary if successful.
    
    Args:
        assembly_code: The assembly code to compile
        
    Returns:
        tuple: (success, binary_data or error_message)
    """
    assembly_code = assembly_code.replace('```assembly\n', '').replace('```', '')
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Write assembly to file
            asm_file = os.path.join(temp_dir, "test.s")
            with open(asm_file, 'w') as f:
                f.write(assembly_code)
            
            # Compile the assembly with -lm to match compilation used elsewhere
            bin_file = os.path.join(temp_dir, "test.bin")
            compile_cmd = f"gcc {asm_file} -o {bin_file} -lm"
            
            compile_result = run_with_timeout(compile_cmd, timeout=30)
            if compile_result['returncode'] != 0:
                return False, f"Compilation error: {compile_result['stderr']}"
            
            # Read the binary
            with open(bin_file, 'rb') as f:
                binary_data = f.read()
            
            return True, binary_data
    except Exception as e:
        return False, f"Error compiling assembly: {str(e)}"


def is_same_assembly(gen_assembly, unopt_assembly):
    # Clean both assembly codes to ensure fair comparison
    gen_assembly = gen_assembly.replace('```assembly\n', '').replace('```', '').strip()
    unopt_assembly = unopt_assembly.replace('```assembly\n', '').replace('```', '').strip()
    
    return gen_assembly == unopt_assembly