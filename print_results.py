#!/usr/bin/env python3
import json
import os
import sys
import math
from pathlib import Path
from typing import Dict, Any
import re

def compute_geometric_mean_speedup(data: Dict[str, Any]) -> tuple:
    speedups = []
    
    # Get the problems from the first key's "problems" field
    first_key = next(iter(data))
    problems_data = data[first_key]["problems"]
    
    # Iterate through all problems
    for problem_id, problem_data in problems_data.items():
        if not isinstance(problem_data, dict):
            continue
            
        # If the problem is correct, use best_speedup; otherwise use 1.0
        speedup = problem_data.get("best_speedup", 1.0) if problem_data.get("overall_correct", False) else 1.0
        speedups.append(speedup)
    
    # Calculate geometric mean and percentiles
    if not speedups:
        return 1.0, 1.0, 1.0, 1.0
    
    # Sort speedups for percentile calculation
    sorted_speedups = sorted(speedups)
    n = len(sorted_speedups)
    
    # Calculate percentiles
    p25 = sorted_speedups[int(n * 0.25)]
    p50 = sorted_speedups[int(n * 0.50)]
    p75 = sorted_speedups[int(n * 0.75)]
    
    # Calculate geometric mean
    log_sum = sum(math.log(s) for s in speedups)
    geometric_mean = math.exp(log_sum / len(speedups))
    
    return geometric_mean, p25, p50, p75

def extract_model_name(file_path):
    """Extract model name from file path and truncate at the B suffix"""
    path = Path(file_path)
    
    # First try to find the parent directory of 'num_iterations_1'
    for parent in path.parents:
        if parent.name.startswith('num_iterations_'):
            model_name = parent.parent.name
            
            # Replace DeepSeek with DS if Distill is in the name
            if 'DeepSeek' in model_name and 'Distill' in model_name:
                model_name = model_name.replace('DeepSeek', 'DS')
            
            # Find position of B that follows a number
            match = re.search(r'\d+B', model_name)
            if match:
                return model_name[:match.end()]
            return model_name
    
    # If we didn't find num_iterations_, look for o4-mini or similar in the path
    for parent in path.parents:
        if parent.name in ['o4-mini', 'o4-base', 'o4-large']:
            return parent.name
    
    # If still not found, return the immediate parent directory name
    return path.parent.name

def process_json_file(file_path):
    """Process a single JSON file and return required metrics"""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        if not isinstance(data, dict) or '0' not in data or 'summary' not in data['0']:
            return None
            
        summary = data['0']['summary']
        if not isinstance(summary, dict):
            return None
            
        # Calculate new speedup statistics
        new_avg_speedup, new_p25, new_p50, new_p75 = compute_geometric_mean_speedup(data)
            
        return {
            'model': extract_model_name(file_path),
            'compilation_rate': summary.get('compilation_rate', 0),
            'accuracy': summary.get('accuracy', 0),  # Don't multiply by 100
            'p25_speedup': max(1.0, new_p25),
            'median_speedup': max(1.0, new_p50),
            'p75_speedup': max(1.0, new_p75),
            'avg_speedup': max(1.0, new_avg_speedup)
        }
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}", file=sys.stderr)
        return None

def format_latex_row(metrics):
    """Format metrics as a LaTeX table row"""
    return (
        f"{metrics['model']} & "
        f"{metrics['compilation_rate']:.1f}\\% & "
        f"{metrics['accuracy']:.1f}\\% & "
        f"{metrics['p25_speedup']:.2f}× & "
        f"{metrics['median_speedup']:.2f}× & "
        f"{metrics['p75_speedup']:.2f}× & "
        f"\\cellcolor{{lightgray}}{metrics['avg_speedup']:.2f}× \\\\"
    )

def format_sheets_row_comma(metrics):
    """Format metrics for Google Sheets using comma separator"""
    return (
        f"{metrics['model']},,,"
        f"{metrics['compilation_rate']:.1f}%%%,,,"
        f"{metrics['accuracy']:.1f}%%%,,,"
        f"{metrics['p25_speedup']:.2f}×,,,"
        f"{metrics['median_speedup']:.2f}×,,,"
        f"{metrics['p75_speedup']:.2f}×,,,"
        f"{metrics['avg_speedup']:.2f}×"
    )

def main():
    if len(sys.argv) != 2:
        print("Usage: python print_latex.py <target_directory>")
        sys.exit(1)
    
    target_dir = sys.argv[1]
    if not os.path.isdir(target_dir):
        print(f"Error: {target_dir} is not a valid directory")
        sys.exit(1)
    
    # Collect all JSON files
    all_metrics = []
    for root, _, files in os.walk(target_dir):
        if 'problem_results.json' in files:
            file_path = os.path.join(root, 'problem_results.json')
            metrics = process_json_file(file_path)
            if metrics:
                all_metrics.append(metrics)
    
    if not all_metrics:
        print("No valid results found!", file=sys.stderr)
        sys.exit(1)
    
    # Sort by average speedup (ascending order), then by compilation rate (ascending order) when speedups are equal
    all_metrics.sort(key=lambda x: (x['avg_speedup'], x['compilation_rate']))
    
    # Print LaTeX table rows with num of rows
    print(f"% In total {len(all_metrics)} rows")
    for metrics in all_metrics:
        print(format_latex_row(metrics))
    
    # print("\n% Google Sheets format (,,, separator):")
    # for metrics in all_metrics:
    #     print(format_sheets_row_comma(metrics))

if __name__ == "__main__":
    main() 