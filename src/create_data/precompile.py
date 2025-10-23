import os
import tempfile
import subprocess
import json
import psutil
from tqdm import tqdm
from datasets import load_dataset
import fire


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


def compile_assembly_code(assembly_code):
    """
    Compile the assembly code and return the compiled binary if successful.
    
    Args:
        assembly_code: The assembly code to compile
        
    Returns:
        dict: Containing binary data if successful, or error information if not
    """
    # Clean up the assembly code (remove markdown code blocks if present)
    if '```assembly' in assembly_code:
        assembly_code = assembly_code[assembly_code.rfind('```assembly') + len('```assembly'):]
    if '```' in assembly_code:
        assembly_code = assembly_code[:assembly_code.rfind('```')]
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Write assembly to file
            asm_file = os.path.join(temp_dir, "test.s")
            with open(asm_file, 'w') as f:
                f.write(assembly_code)
            
            # Compile the assembly with -lm to match compilation in evaluate.py
            bin_file = os.path.join(temp_dir, "test.bin")
            compile_cmd = f"gcc {asm_file} -o {bin_file} -lm"
            
            compile_result = run_with_timeout(compile_cmd, timeout=30)
            
            if compile_result['returncode'] == 0:
                # Read the binary data
                with open(bin_file, 'rb') as f:
                    binary_data = f.read()
                return {
                    'success': True,
                    'binary': binary_data,
                    'stderr': compile_result['stderr']
                }
            else:
                return {
                    'success': False,
                    'binary': None,
                    'stderr': compile_result['stderr']
                }
    
    except Exception as e:
        return {
            'success': False,
            'binary': None,
            'stderr': str(e)
        }


def precompile_dataset(ds_name,  cache_dir=None):
    """
    Precompile the unoptimized assembly code in a dataset and add compiled binaries.
    
    Args:
        ds_name: The name of the dataset
        split: The dataset split to process (default: 'train')
        cache_dir: Optional cache directory for HuggingFace datasets
    
    Returns:
        The processed dataset with a new 'unoptimized_compiled' column containing the binary
    """
    # Set default cache dir if not provided
    if cache_dir is None and 'HF_CACHE' in os.environ:
        cache_dir = os.environ['HF_CACHE']

    dataset = load_dataset(ds_name, cache_dir=cache_dir)
    
    # Function to process a single example
    def process_example(example):
        unoptimized_assembly = example['extra_info']['unoptimized_assembly'].replace('```assembly\n', '').replace('```', '')
        compilation_result = compile_assembly_code(unoptimized_assembly)
        
        # Store the binary or error message
        assert compilation_result['success']
        example['extra_info']['unoptimized_compiled'] = compilation_result['binary']
        
        return example
    
    # Process all examples in the dataset with a progress bar
    print(f"Compiling unoptimized assembly code for {len(dataset)} examples...")
    processed_dataset = dataset.map(
        process_example,
        desc="Compiling assembly",
        num_proc=1  # Process one at a time to avoid potential resource issues
    )
    processed_dataset.push_to_hub(ds_name)



def main(ds_name):
    """Main function to precompile a dataset."""
    # Set cache dir from environment variable if available
    cache_dir = os.environ.get('HF_CACHE', None)
    
    # Process the dataset
    precompile_dataset(ds_name,  cache_dir)
    



if __name__ == "__main__":
    fire.Fire(main)
