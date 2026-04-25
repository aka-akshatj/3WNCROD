"""
Runner script for PyOD-based outlier detection
Processes all datasets in the Datasets folder using multiple PyOD algorithms
"""

import numpy as np
import time
import os
import sys
from datetime import datetime

# Import PyOD detection module
# Adjust import path if needed based on your directory structure
try:
    from pyod_detection import (
        process_dataset_with_multiple_algorithms,
        get_top_anomalies
    )
except ImportError:
    # If running from parent directory
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from pyod_detection import (
        process_dataset_with_multiple_algorithms,
        get_top_anomalies
    )

# List of available datasets
datasets = [
    'annthyroid',
    'creditA_plus_42_variant1',
    'german_1_14_variant1',
    'heart270_2_16_variant1',
    'lymphography',
    'mammography',
    'mushroom_p_221_variant1',
    'musk',
    'thyroid',
    'wdbc_M_39_variant1'
]

# Algorithms to use (you can modify this list)
algorithms = [
    'IForest',    # Isolation Forest
    'LOF',        # Local Outlier Factor
    'COPOD',      # Copula-based Outlier Detection
    'HBOS',       # Histogram-based Outlier Score
    'KNN',        # k-Nearest Neighbors
    'AutoEncoder'
]

# algorithms = ['AutoEncoder']

# Optional: Use fewer algorithms for faster testing
# algorithms = ['IForest', 'COPOD', 'HBOS']


class Tee:
    """Class to write to both file and console simultaneously"""
    def __init__(self, *files):
        self.files = files
    
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()  # Ensure immediate write
    
    def flush(self):
        for f in self.files:
            f.flush()


def print_results_summary(results, dataset_name):
    """Print detailed results for a dataset"""
    print(f"\n{'='*60}")
    print(f"Results for: {dataset_name}")
    print(f"{'='*60}")
    print(f"Data shape: {results['data_shape']}")
    
    for algo_name, algo_results in results['algorithms'].items():
        if 'error' in algo_results:
            print(f"\n{algo_name}: ERROR - {algo_results['error']}")
            continue
        
        print(f"\n{algo_name}:")
        print(f"  Fit time: {algo_results['fit_time']:.2f} seconds ({algo_results['fit_time']/60:.2f} minutes)")
        print(f"  Outliers detected: {algo_results['n_outliers']} ({algo_results['outlier_rate']*100:.2f}%)")
        print(f"  Score statistics:")
        print(f"    Min: {algo_results['min_score']:.6f}")
        print(f"    Max: {algo_results['max_score']:.6f}")
        print(f"    Mean: {algo_results['mean_score']:.6f}")
        print(f"    Std: {algo_results['std_score']:.6f}")
        
        # Top anomalies
        top_k = min(10, len(algo_results['outlier_scores']))
        top_indices, top_scores = get_top_anomalies(
            algo_results['outlier_scores'], 
            top_k=top_k
        )
        print(f"  Top {top_k} anomalies:")
        for idx, score in zip(top_indices, top_scores):
            print(f"    Sample {idx}: {score:.6f}")


def main():
    """Main function to process all datasets"""
    # Create output directory if it doesn't exist
    output_dir = 'output_logs'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create timestamped output file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file_path = os.path.join(output_dir, f'pyod_output_{timestamp}.txt')
    
    # Open file for writing
    log_file = open(output_file_path, 'w', encoding='utf-8')
    
    # Store original stdout for restoration
    original_stdout = sys.stdout
    
    # Create Tee object to write to both console and file
    tee = Tee(sys.stdout, log_file)
    sys.stdout = tee
    
    try:
        # Track total runtime
        total_start_time = time.time()
        
        # Store all results
        all_results = []
        timing_summary = []
        
        print("="*60)
        print("PyOD Outlier Detection - Batch Processing")
        print("="*60)
        print(f"Algorithms: {', '.join(algorithms)}")
        print(f"Datasets: {len(datasets)}")
        print(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Output log: {output_file_path}")
        print("="*60)
        
        # Process each dataset
        for idx, dataset_name in enumerate(datasets, 1):
            dataset_start_time = time.time()
            
            print(f"\n{'='*60}")
            print(f"Processing dataset {idx}/{len(datasets)}: {dataset_name}")
            print(f"Started at: {time.strftime('%H:%M:%S')}")
            print(f"{'='*60}")
            
            try:
                # Try to find dataset (.mat or .csv)
                dataset_path_mat = f'./Datasets/{dataset_name}.mat'
                dataset_path_csv = f'./Datasets/{dataset_name}.csv'
                
                if os.path.exists(dataset_path_mat):
                    dataset_path = dataset_path_mat
                elif os.path.exists(dataset_path_csv):
                    dataset_path = dataset_path_csv
                else:
                    print(f"ERROR: File not found: {dataset_path_mat} or {dataset_path_csv}")
                    timing_summary.append({
                        'dataset': dataset_name,
                        'status': 'file_not_found',
                        'time': 0
                    })
                    continue
                
                # Process dataset
                results = process_dataset_with_multiple_algorithms(
                    dataset_path,
                    dataset_name,
                    algorithms=algorithms
                )
                
                # Calculate timing
                dataset_time = time.time() - dataset_start_time
                
                # Store results
                all_results.append(results)
                timing_summary.append({
                    'dataset': dataset_name,
                    'status': 'success',
                    'time': dataset_time,
                    'shape': results['data_shape']
                })
                
                # Print detailed results
                print_results_summary(results, dataset_name)
                
                print(f"\n{'='*60}")
                print(f"Completed {dataset_name} in {dataset_time:.2f} seconds ({dataset_time/60:.2f} minutes)")
                print(f"{'='*60}")
                
            except Exception as e:
                dataset_time = time.time() - dataset_start_time
                print(f"\nERROR processing {dataset_name}: {str(e)}")
                timing_summary.append({
                    'dataset': dataset_name,
                    'status': 'error',
                    'error': str(e),
                    'time': dataset_time
                })
        
        # Final summary
        total_time = time.time() - total_start_time
        
        print(f"\n\n{'='*60}")
        print("FINAL SUMMARY")
        print(f"{'='*60}")
        print(f"Total runtime: {total_time:.2f} seconds ({total_time/60:.2f} minutes, {total_time/3600:.2f} hours)")
        print(f"\nPer-dataset timing:")
        print(f"{'Dataset':<30} {'Status':<15} {'Time (min)':<15} {'Shape':<20}")
        print(f"{'-'*80}")
        
        for timing in timing_summary:
            status = timing['status']
            time_str = f"{timing['time']/60:.2f}" if timing['time'] > 0 else "N/A"
            shape_str = str(timing.get('shape', 'N/A'))
            print(f"{timing['dataset']:<30} {status:<15} {time_str:<15} {shape_str:<20}")
        
        # Algorithm performance summary
        print(f"\n{'='*60}")
        print("Algorithm Performance Summary")
        print(f"{'='*60}")
        
        algo_times = {algo: [] for algo in algorithms}
        for result in all_results:
            for algo_name, algo_results in result['algorithms'].items():
                if 'fit_time' in algo_results:
                    algo_times[algo_name].append(algo_results['fit_time'])
        
        print(f"{'Algorithm':<20} {'Avg Time (s)':<15} {'Total Time (s)':<15} {'Runs':<10}")
        print(f"{'-'*60}")
        for algo in algorithms:
            if algo_times[algo]:
                avg_time = np.mean(algo_times[algo])
                total_time_algo = np.sum(algo_times[algo])
                n_runs = len(algo_times[algo])
                print(f"{algo:<20} {avg_time:<15.2f} {total_time_algo:<15.2f} {n_runs:<10}")
        
        print(f"\n{'='*60}")
        print(f"Completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")
        print(f"\nAll output saved to: {output_file_path}")
        print(f"{'='*60}")
    
    finally:
        # Restore original stdout and close file
        sys.stdout = original_stdout
        log_file.close()
        print(f"\nOutput log saved to: {output_file_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        # Restore stdout if it was redirected
        if sys.stdout != sys.__stdout__:
            sys.stdout = sys.__stdout__
        print("\n\nProcess interrupted by user.")
        sys.exit(0)
    except Exception as e:
        # Restore stdout if it was redirected
        if sys.stdout != sys.__stdout__:
            sys.stdout = sys.__stdout__
        print(f"\n\nFatal error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

