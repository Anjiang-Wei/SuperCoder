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

def calculate_metrics(model_name, ds_name, best_of, gen_results_by_problem, test_dataset, results_by_problem_and_sample, compilation_failures, identical_assembly_samples):
    # Calculate metrics for each sample of each problem
    sample_metrics = {}
    for (problem_idx, sample_id), results in results_by_problem_and_sample.items():
        # Calculate number of successful test cases
        successful_test_cases = [r for r in results if r['status'] == 'success']
        num_successful = len(successful_test_cases)
        num_total = len(results)
        
        # Check if this sample has identical assembly to unoptimized
        is_identical = identical_assembly_samples.get((problem_idx, sample_id), False)
        is_fully_correct = num_successful == num_total
        
        # Calculate average speedup for successful test cases
        if is_identical or not is_fully_correct:
            # If identical to unoptimized or not fully correct, set speedup to 1.0
            avg_speedup = 1.0
        else:
            # Only use actual speedups for fully correct implementations
            successful_speedups = [r['speedup'] for r in successful_test_cases]
            avg_speedup = np.mean(successful_speedups) if successful_speedups else 1.0  # Default to 1.0
        
        sample_metrics[(problem_idx, sample_id)] = {
            'num_successful': num_successful,
            'num_total': num_total,
            'is_fully_correct': is_fully_correct,
            'avg_speedup': avg_speedup,
            'results': results,
            'identical_to_unoptimized': is_identical  # NEW
        }
    
    # Select the best sample for each problem
    best_samples = {}
    for problem_idx in range(len(test_dataset)):
        problem_samples = [(k, v) for k, v in sample_metrics.items() if k[0] == problem_idx]
        if not problem_samples:
            continue
        
        # Sort by number of successful test cases (descending) and then by speedup (descending)
        problem_samples.sort(key=lambda x: (x[1]['num_successful'], x[1]['avg_speedup']), reverse=True)
        
        # Get the best sample
        best_key, best_metrics = problem_samples[0]
        _, best_sample_id = best_key
        
        # Find the generated assembly code for this sample
        best_gen_res = next(s['output'] for s in gen_results_by_problem[problem_idx] 
                           if s['sample_id'] == best_sample_id)
        
        # Store the best sample
        best_samples[problem_idx] = {
            'correct': best_metrics['is_fully_correct'],
            'speedup': best_metrics['avg_speedup'],
            'test_cases': best_metrics['results'],
            'num_test_cases': best_metrics['num_total'],
            'generated_assembly': best_gen_res,
            'unoptimized_assembly': test_dataset[problem_idx]['extra_info']['unoptimized_assembly'],
            'num_successful': best_metrics['num_successful'],
            'best_sample_id': best_sample_id,
            'identical_to_unoptimized': best_metrics['identical_to_unoptimized']  # NEW
        }
    
    # Calculate final statistics using the best samples
    total_problems = len(test_dataset)
    correct_problems = sum(1 for p in best_samples.values() if p['correct'])
    compiled_problems = total_problems - len(compilation_failures)
    accuracy = (correct_problems / total_problems) * 100 if total_problems > 0 else 0
    compilation_rate = (compiled_problems / total_problems) * 100 if total_problems > 0 else 0
    
    # Ensure that we only count speedup for correct implementations
    all_speedups = [p['speedup'] if p['correct'] else 1.0 for p in best_samples.values()]
    
    print(f"Compilation rate: {compilation_rate:.2f}% ({compiled_problems}/{total_problems})")
    print(f"Accuracy: {accuracy:.2f}% ({correct_problems}/{total_problems})")
    
    if all_speedups:
        avg_speedup = np.mean(all_speedups)
        median_speedup = np.median(all_speedups)
        max_speedup = np.max(all_speedups)
        # Calculate additional statistics
        p25_speedup = np.percentile(all_speedups, 25)
        p75_speedup = np.percentile(all_speedups, 75)
        std_speedup = np.std(all_speedups)
        print(f"Average speedup: {avg_speedup:.2f}x")
        print(f"Median speedup: {median_speedup:.2f}x")
        print(f"Maximum speedup: {max_speedup:.2f}x")
        print(f"25th percentile speedup: {p25_speedup:.2f}x")
        print(f"75th percentile speedup: {p75_speedup:.2f}x")
        print(f"Standard deviation of speedup: {std_speedup:.2f}x")
    
    # Create a new structure that includes all samples
    all_samples_results = {}
    for problem_idx in range(len(test_dataset)):
        problem_samples = [(k, v) for k, v in sample_metrics.items() if k[0] == problem_idx]
        
        # Handle compilation failures
        if problem_idx in compilation_failures:
            all_samples_results[problem_idx] = {
                'compilation_failed': True,
                'unoptimized_assembly': test_dataset[problem_idx]['extra_info']['unoptimized_assembly'],
                'samples': {},
                'overall_correct': False,
                'best_speedup': 1.0  # Default to 1.0 for compilation failures
            }
            continue
            
        if not problem_samples:
            continue
            
        # Store all samples with their metrics
        samples_dict = {}
        for key_metrics_pair in problem_samples:
            key, metrics = key_metrics_pair
            prob_idx, sample_id = key
            
            # Find the generated assembly code for this sample
            gen_res = next(s['output'] for s in gen_results_by_problem[problem_idx] 
                          if s['sample_id'] == sample_id)
            
            # Mark if this is the best sample
            is_best = (sample_id == best_samples[problem_idx]['best_sample_id']) if problem_idx in best_samples else False
            
            samples_dict[str(sample_id)] = {  # Convert to string key for JSON
                'is_best': is_best,
                'correct': metrics['is_fully_correct'],
                'speedup': metrics['avg_speedup'],
                'test_cases': metrics['results'],
                'num_test_cases': metrics['num_total'],
                'generated_assembly': gen_res,
                'num_successful': metrics['num_successful'],
                'identical_to_unoptimized': metrics['identical_to_unoptimized']  # NEW
            }
        
        # Store all samples along with problem-level information
        best_sample_id = best_samples[problem_idx]['best_sample_id'] if problem_idx in best_samples else None
        overall_correct = best_samples[problem_idx]['correct'] if problem_idx in best_samples else False
        best_speedup = best_samples[problem_idx]['speedup'] if problem_idx in best_samples else 1.0  # Default to 1.0
        
        all_samples_results[problem_idx] = {
            'compilation_failed': False,
            'samples': samples_dict,
            'unoptimized_assembly': test_dataset[problem_idx]['extra_info']['unoptimized_assembly'],
            'best_sample_id': best_sample_id,
            'overall_correct': overall_correct,
            'best_speedup': best_speedup
        }
    

    
    # Add summary statistics to make it easier to analyze results
    summary = {
        'total_problems': total_problems,
        'correct_problems': correct_problems,
        'compiled_problems': compiled_problems,
        'compilation_failures': len(compilation_failures),
        'compilation_rate': compilation_rate,
        'accuracy': accuracy,
        'avg_speedup': float(avg_speedup) if 'avg_speedup' in locals() else 0.0,
        'median_speedup': float(median_speedup) if 'median_speedup' in locals() else 0.0,
        'max_speedup': float(max_speedup) if 'max_speedup' in locals() else 0.0,
        'p25_speedup': float(p25_speedup) if 'p25_speedup' in locals() else 0.0,
        'p75_speedup': float(p75_speedup) if 'p75_speedup' in locals() else 0.0,
        'std_speedup': float(std_speedup) if 'std_speedup' in locals() else 0.0,
        'best_of': best_of,
        'model': model_name,
        'dataset': ds_name
    }
    
    # Add summary to the results
    output_data = {
        'summary': summary,
        'problems': all_samples_results
    }
    
    return output_data

def compute_summary_from_file(metrics_file_path):
    """
    Compute and print summary statistics from an existing metrics JSON file.
    
    Args:
        metrics_file_path: Path to the metrics JSON file
        
    Returns:
        dict: Summary statistics dictionary
    """
    print(f"Loading metrics from: {metrics_file_path}")
    
    try:
        with open(metrics_file_path, 'r') as f:
            data = json.load(f)
        
        # Check if the file already has a summary section
        if 'summary' in data:
            summary = data['summary']
            print("Found existing summary section in the file.")
        else:
            # If no summary exists, we need to compute it
            print("No summary found. Computing from problem data...")
            problems = data.get('problems', {})
            
            # Count total problems, correct problems, and compile failures
            total_problems = len(problems)
            correct_problems = sum(1 for _, p in problems.items() if not p.get('compilation_failed', False) and p.get('overall_correct', False))
            compilation_failures = sum(1 for _, p in problems.items() if p.get('compilation_failed', False))
            compiled_problems = total_problems - compilation_failures
            
            # Calculate accuracy and compilation rate
            accuracy = (correct_problems / total_problems) * 100 if total_problems > 0 else 0
            compilation_rate = (compiled_problems / total_problems) * 100 if total_problems > 0 else 0
            
            # Extract speedups for correct problems only
            all_speedups = []
            for problem_id, problem in problems.items():
                if not problem.get('compilation_failed', False):
                    if problem.get('overall_correct', False):
                        all_speedups.append(problem.get('best_speedup', 1.0))
                    else:
                        # For incorrect solutions, use speedup of 1.0
                        all_speedups.append(1.0)
            
            # Create summary dictionary
            summary = {
                'total_problems': total_problems,
                'correct_problems': correct_problems,
                'compiled_problems': compiled_problems,
                'compilation_failures': compilation_failures,
                'compilation_rate': compilation_rate,
                'accuracy': accuracy
            }
            
            # Add dataset/model info if available in the original data
            if 'model' in data.get('summary', {}):
                summary['model'] = data['summary']['model']
            if 'dataset' in data.get('summary', {}):
                summary['dataset'] = data['summary']['dataset']
            if 'best_of' in data.get('summary', {}):
                summary['best_of'] = data['summary']['best_of']
        
        # Compute or retrieve speedup statistics
        if 'problems' in data:
            # Get speedups directly from problem data
            all_speedups = []
            for problem_id, problem in data['problems'].items():
                if not problem.get('compilation_failed', False):
                    if problem.get('overall_correct', False):
                        all_speedups.append(problem.get('best_speedup', 1.0))
                    else:
                        # For incorrect solutions, use speedup of 1.0
                        all_speedups.append(1.0)
        else:
            # Try to use existing speedup data if available
            all_speedups = []
            for speedup_key in ['avg_speedup', 'median_speedup', 'max_speedup']:
                if speedup_key in summary:
                    all_speedups.append(summary[speedup_key])
            
            if not all_speedups:
                print("Warning: No speedup data found in the JSON file.")
                all_speedups = [1.0]  # Default if no data available
        
        # Calculate speedup statistics
        if all_speedups:
            avg_speedup = np.mean(all_speedups)
            median_speedup = np.median(all_speedups)
            max_speedup = np.max(all_speedups)
            p25_speedup = np.percentile(all_speedups, 25)
            p75_speedup = np.percentile(all_speedups, 75)
            std_speedup = np.std(all_speedups)
            
            # Update or add speedup metrics to summary
            summary.update({
                'avg_speedup': float(avg_speedup),
                'median_speedup': float(median_speedup),
                'max_speedup': float(max_speedup),
                'p25_speedup': float(p25_speedup),
                'p75_speedup': float(p75_speedup),
                'std_speedup': float(std_speedup)
            })
        
        # Print the summary statistics
        print(f"\nSummary Statistics:")
        print(f"Compilation rate: {summary['compilation_rate']:.2f}% ({summary['compiled_problems']}/{summary['total_problems']})")
        print(f"Accuracy: {summary['accuracy']:.2f}% ({summary['correct_problems']}/{summary['total_problems']})")
        
        if 'avg_speedup' in summary:
            print(f"Average speedup: {summary['avg_speedup']:.2f}x")
            print(f"Median speedup: {summary['median_speedup']:.2f}x")
            print(f"Maximum speedup: {summary['max_speedup']:.2f}x")
            print(f"25th percentile speedup: {summary['p25_speedup']:.2f}x")
            print(f"75th percentile speedup: {summary['p75_speedup']:.2f}x")
            print(f"Standard deviation of speedup: {summary['std_speedup']:.2f}x")
        
        return summary
    
    except Exception as e:
        print(f"Error processing metrics file: {str(e)}")
        return None

def main():
    """CLI entry point for metrics functions"""
    fire.Fire({
        'compute_summary': compute_summary_from_file
    })

if __name__ == "__main__":
    main()