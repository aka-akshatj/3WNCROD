"""
Generate comprehensive comparison report with all metrics including ROC/AUC
Reads PyOD output logs and generates a detailed comparison report
"""

import os
import re
import json
import numpy as np
from datetime import datetime
from pathlib import Path
import pickle

def extract_metrics_from_log(log_file_path):
    """Extract metrics from PyOD output log file"""
    
    with open(log_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    results = {}
    
    # Split by "Processing dataset"
    sections = re.split(r'Processing dataset \d+/\d+: ', content)
    
    for section in sections[1:]:  # Skip first empty split
        lines = section.split('\n')
        
        # First line after split is dataset name
        dataset_name = lines[0].strip()
        
        results[dataset_name] = {'algorithms': {}}
        
        # Extract data shape
        for line in lines:
            if 'Data shape:' in line:
                match = re.search(r'Data shape: \((\d+), (\d+)\)', line)
                if match:
                    results[dataset_name]['shape'] = (int(match.group(1)), int(match.group(2)))
                break
        
        # Extract algorithm results - look for "Algorithm:" pattern
        current_algo = None
        current_metrics = {}
        
        for line in lines:
            line = line.strip()
            
            # Check for algorithm name (starts with capital, ends with colon, appears at start of metrics block)
            if line and line.endswith(':') and not line.startswith(('Data', 'Loading', 'Labels', 'Ground', 'Started', 'Completed', 'Results', 'Top', 'Sample', 'Score', 'Min:', 'Max:', 'Mean:', 'Std:', 'Running')):
                if any(x in line for x in ['IForest', 'LOF', 'COPOD', 'HBOS', 'KNN', 'AutoEncoder']):
                    # Save previous algo if exists
                    if current_algo and current_metrics:
                        results[dataset_name]['algorithms'][current_algo] = current_metrics
                    
                    # Start new algorithm
                    current_algo = line.rstrip(':')
                    current_metrics = {}
            
            # Extract metrics
            if current_algo:
                if 'Fit time:' in line:
                    match = re.search(r'Fit time: ([\d.]+) seconds', line)
                    if match:
                        current_metrics['fit_time'] = float(match.group(1))
                
                elif 'Detected' in line and 'outliers' in line:
                    match = re.search(r'Detected (\d+) outliers \(([\d.]+)%\)', line)
                    if match:
                        current_metrics['n_outliers'] = int(match.group(1))
                        current_metrics['outlier_rate'] = float(match.group(2)) / 100
                
                elif 'ROC-AUC:' in line:
                    match = re.search(r'ROC-AUC: ([\d.]+)', line)
                    if match:
                        current_metrics['roc_auc'] = float(match.group(1))
        
        # Save last algorithm
        if current_algo and current_metrics:
            results[dataset_name]['algorithms'][current_algo] = current_metrics
    
    return results


def save_results_as_json(results, output_path):
    """Save results in JSON format for future reference"""
    
    # Convert numpy arrays/types to native Python types
    def convert_to_serializable(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(item) for item in obj]
        return obj
    
    with open(output_path, 'w') as f:
        json.dump(convert_to_serializable(results), f, indent=2)
    
    print(f"Results saved as JSON: {output_path}")


def generate_comparison_report(results_dict, output_file='output_logs/COMPARISON_REPORT.md'):
    """Generate comprehensive markdown comparison report"""
    
    datasets = sorted(results_dict.keys())
    algorithms = set()
    
    # Collect all algorithms
    for dataset in datasets:
        algorithms.update(results_dict[dataset].get('algorithms', {}).keys())
    algorithms = sorted(list(algorithms))
    
    report = []
    report.append("# Outlier Detection Methods Comparison Report\n")
    report.append("## PyOD Algorithms Performance Analysis\n")
    report.append(f"**Report Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report.append(f"**Datasets Analyzed:** {len(datasets)}\n")
    report.append(f"**Methods Compared:** {', '.join(algorithms)}\n")
    report.append("---\n")
    
    # Calculate statistics
    algo_stats = {algo: {
        'times': [],
        'outlier_rates': [],
        'roc_aucs': []
    } for algo in algorithms}
    
    for dataset in datasets:
        for algo in algorithms:
            if algo in results_dict[dataset].get('algorithms', {}):
                algo_data = results_dict[dataset]['algorithms'][algo]
                algo_stats[algo]['times'].append(algo_data.get('fit_time', 0))
                algo_stats[algo]['outlier_rates'].append(algo_data.get('outlier_rate', 0))
                if 'roc_auc' in algo_data:
                    algo_stats[algo]['roc_aucs'].append(algo_data['roc_auc'])
    
    # Performance Overview
    report.append("\n## 1. Algorithm Performance Overview\n")
    
    perf_table = []
    perf_table.append("| Algorithm | Avg Time (s) | Total Time (s) | Avg Outlier Rate | Avg ROC-AUC | Runs |")
    perf_table.append("|-----------|--------------|----------------|------------------|-------------|------|")
    
    for algo in algorithms:
        times = algo_stats[algo]['times']
        roc_aucs = algo_stats[algo]['roc_aucs']
        outlier_rates = algo_stats[algo]['outlier_rates']
        
        if times:
            avg_time = np.mean(times)
            total_time = np.sum(times)
            avg_rate = np.mean(outlier_rates) * 100 if outlier_rates else 0
            avg_auc = np.mean(roc_aucs) if roc_aucs else float('nan')
            n_runs = len(times)
            
            auc_str = f"{avg_auc:.4f}" if not np.isnan(avg_auc) else "N/A"
            perf_table.append(
                f"| {algo} | {avg_time:.4f} | {total_time:.4f} | {avg_rate:.2f}% | {auc_str} | {n_runs} |"
            )
    
    report.extend(perf_table)
    report.append("")
    
    # Per-dataset results
    report.append("\n## 2. Per-Dataset Results\n")
    
    for dataset in datasets:
        report.append(f"\n### Dataset: {dataset}\n")
        
        if 'shape' in results_dict[dataset]:
            shape = results_dict[dataset]['shape']
            report.append(f"- **Shape:** {shape[0]} samples × {shape[1]} features\n")
        
        dataset_algos = results_dict[dataset].get('algorithms', {})
        
        if dataset_algos:
            dataset_table = []
            dataset_table.append("| Algorithm | Time (s) | Outliers Detected | Rate | ROC-AUC |")
            dataset_table.append("|-----------|----------|------------------|------|---------|")
            
            for algo in sorted(dataset_algos.keys()):
                algo_data = dataset_algos[algo]
                time_str = f"{algo_data.get('fit_time', 0):.4f}"
                n_outliers = algo_data.get('n_outliers', 'N/A')
                rate_str = f"{algo_data.get('outlier_rate', 0)*100:.2f}%" if 'outlier_rate' in algo_data else "N/A"
                
                roc_auc = algo_data.get('roc_auc')
                auc_str = f"{roc_auc:.4f}" if roc_auc is not None else "N/A"
                
                dataset_table.append(
                    f"| {algo} | {time_str} | {n_outliers} | {rate_str} | {auc_str} |"
                )
            
            report.extend(dataset_table)
    
    # Statistical Summary
    report.append("\n## 3. Statistical Summary\n")
    
    summary_table = []
    summary_table.append("| Metric | Min | Max | Mean | Std |")
    summary_table.append("|--------|-----|-----|------|-----|")
    
    for algo in algorithms:
        times = algo_stats[algo]['times']
        if times:
            times_array = np.array(times)
            summary_table.append(
                f"| {algo} Time (s) | {np.min(times_array):.4f} | {np.max(times_array):.4f} | "
                f"{np.mean(times_array):.4f} | {np.std(times_array):.4f} |"
            )
        
        roc_aucs = algo_stats[algo]['roc_aucs']
        if roc_aucs:
            auc_array = np.array(roc_aucs)
            summary_table.append(
                f"| {algo} ROC-AUC | {np.min(auc_array):.4f} | {np.max(auc_array):.4f} | "
                f"{np.mean(auc_array):.4f} | {np.std(auc_array):.4f} |"
            )
    
    report.extend(summary_table)
    
    # Key Findings
    report.append("\n## 4. Key Findings\n")
    
    # Find fastest algorithm
    fastest_algo = min([(algo, np.mean(algo_stats[algo]['times'])) 
                        for algo in algorithms if algo_stats[algo]['times']], 
                       key=lambda x: x[1])[0]
    report.append(f"- **Fastest Algorithm:** {fastest_algo}\n")
    
    # Find best ROC-AUC
    best_auc_algo = max([(algo, np.mean(algo_stats[algo]['roc_aucs'])) 
                         for algo in algorithms if algo_stats[algo]['roc_aucs']], 
                        key=lambda x: x[1] if not np.isnan(x[1]) else -1, 
                        default=(None, 0))
    if best_auc_algo[0]:
        report.append(f"- **Best Average ROC-AUC:** {best_auc_algo[0]} ({best_auc_algo[1]:.4f})\n")
    
    # Consistency
    consistency_scores = {
        algo: np.std(algo_stats[algo]['outlier_rates']) 
        for algo in algorithms if algo_stats[algo]['outlier_rates']
    }
    if consistency_scores:
        most_consistent = min(consistency_scores, key=consistency_scores.get)
        report.append(f"- **Most Consistent:** {most_consistent} (outlier rate std: {consistency_scores[most_consistent]:.4f})\n")
    
    # Write report
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    print(f"\nComparison report generated: {output_file}")
    return '\n'.join(report)


def main():
    """Main function to generate reports"""
    
    # Find the latest PyOD output log
    output_logs_dir = 'output_logs'
    
    if not os.path.exists(output_logs_dir):
        print(f"Error: {output_logs_dir} directory not found")
        return
    
    # Get all pyod output files
    pyod_files = sorted([f for f in os.listdir(output_logs_dir) 
                        if f.startswith('pyod_output_') and f.endswith('.txt')])
    
    if not pyod_files:
        print(f"No PyOD output files found in {output_logs_dir}")
        return
    
    latest_log = os.path.join(output_logs_dir, pyod_files[-1])
    print(f"Reading latest PyOD output: {latest_log}")
    
    # Extract metrics
    results = extract_metrics_from_log(latest_log)
    
    # Save as JSON
    json_output = os.path.join(output_logs_dir, 'comparison_results.json')
    save_results_as_json(results, json_output)
    
    # Generate report
    report_output = os.path.join(output_logs_dir, 'COMPARISON_REPORT.md')
    generate_comparison_report(results, report_output)
    
    print("\n✓ Comparison report generated successfully!")


if __name__ == "__main__":
    main()
