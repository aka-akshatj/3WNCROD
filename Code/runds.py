import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import MinMaxScaler
import os
import sys
import time
from datetime import datetime

# Import the WNCROD function
from WNCROD import WNCROD


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

# WNCROD parameters
lammda = 1  # Radius adjustment parameter


def main():
    """Main function to process all datasets with WNCROD"""
    # Create output directory if it doesn't exist
    output_dir = 'output_logs'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create timestamped output file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file_path = os.path.join(output_dir, f'wncrod_output_{timestamp}.txt')
    
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
        
        # Store all results and timing
        all_results = []
        timing_summary = []
        
        print("="*60)
        print("WNCROD Outlier Detection - Batch Processing")
        print("="*60)
        print(f"Lambda parameter: {lammda}")
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
                # Load dataset
                load_start = time.time()
                dataset_path = f'./Datasets/{dataset_name}.mat'
                
                # Check if file exists
                if not os.path.exists(dataset_path):
                    print(f"ERROR: File not found: {dataset_path}")
                    timing_summary.append({
                        'dataset': dataset_name,
                        'status': 'file_not_found',
                        'time': 0
                    })
                    continue
                
                load_data = loadmat(dataset_path)
                
                # Get the data (variable name is usually the dataset name)
                data_key = [k for k in load_data.keys() if not k.startswith('__')][0]
                trandata = load_data[data_key]
                load_time = time.time() - load_start
                
                print(f"Data shape: {trandata.shape}")
                print(f"Load time: {load_time:.2f} seconds")
                
                # Normalize numerical columns to [0,1]
                norm_start = time.time()
                scaler = MinMaxScaler()
                trandata_normalized = scaler.fit_transform(trandata)
                norm_time = time.time() - norm_start
                print(f"Normalization time: {norm_time:.2f} seconds")
                
                # Select sample indices (X_tem)
                n_samples = trandata_normalized.shape[0]
                X_tem = list(range(n_samples))  # Use all samples
                
                # Or use a subset (e.g., first 50% of samples) for faster testing
                # X_tem = list(range(0, min(1000, n_samples)))  # Use max 1000 samples
                
                print(f"Using {len(X_tem)} samples for WNCROD computation")
                
                # Run WNCROD
                print(f"Starting WNCROD computation...")
                wncrod_start = time.time()
                out_scores = WNCROD(trandata_normalized, X_tem, lammda)
                wncrod_time = time.time() - wncrod_start
                
                # Calculate total dataset time
                dataset_time = time.time() - dataset_start_time
                
                # Display results
                print(f"\n{'='*60}")
                print(f"Results for {dataset_name}:")
                print(f"{'='*60}")
                print(f"Outlier scores computed: {len(out_scores)} samples")
                print(f"Min score: {np.min(out_scores):.6f}")
                print(f"Max score: {np.max(out_scores):.6f}")
                print(f"Mean score: {np.mean(out_scores):.6f}")
                print(f"Std score: {np.std(out_scores):.6f}")
                
                # Find top anomalies (highest scores)
                top_k = min(10, len(out_scores))
                top_indices = np.argsort(out_scores)[-top_k:][::-1]
                print(f"\nTop {top_k} anomalies (highest scores):")
                for idx_val, score in zip(top_indices, out_scores[top_indices]):
                    print(f"  Sample {idx_val}: {score:.6f}")
                
                # Timing summary
                print(f"\n{'='*60}")
                print(f"Timing Summary for {dataset_name}:")
                print(f"  Load time: {load_time:.2f} seconds")
                print(f"  Normalization time: {norm_time:.2f} seconds")
                print(f"  WNCROD computation: {wncrod_time:.2f} seconds ({wncrod_time/60:.2f} minutes)")
                print(f"  Total dataset time: {dataset_time:.2f} seconds ({dataset_time/60:.2f} minutes)")
                print(f"{'='*60}")
                
                # Store results
                all_results.append({
                    'dataset': dataset_name,
                    'shape': trandata.shape,
                    'outlier_scores': out_scores,
                    'top_anomalies': top_indices[:top_k],
                    'top_scores': out_scores[top_indices[:top_k]]
                })
                
                timing_summary.append({
                    'dataset': dataset_name,
                    'status': 'success',
                    'load_time': load_time,
                    'norm_time': norm_time,
                    'wncrod_time': wncrod_time,
                    'total_time': dataset_time,
                    'shape': trandata.shape
                })
                
            except Exception as e:
                dataset_time = time.time() - dataset_start_time
                print(f"\nERROR processing {dataset_name}: {str(e)}")
                import traceback
                traceback.print_exc()
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
        print(f"{'Dataset':<30} {'Status':<15} {'WNCROD Time (min)':<20} {'Total Time (min)':<20} {'Shape':<20}")
        print(f"{'-'*105}")
        
        for timing in timing_summary:
            status = timing['status']
            if status == 'success':
                wncrod_time_str = f"{timing['wncrod_time']/60:.2f}" if 'wncrod_time' in timing else "N/A"
                total_time_str = f"{timing['total_time']/60:.2f}" if 'total_time' in timing else "N/A"
            else:
                wncrod_time_str = "N/A"
                total_time_str = f"{timing.get('time', 0)/60:.2f}" if timing.get('time', 0) > 0 else "N/A"
            
            shape_str = str(timing.get('shape', 'N/A'))
            print(f"{timing['dataset']:<30} {status:<15} {wncrod_time_str:<20} {total_time_str:<20} {shape_str:<20}")
        
        # Performance statistics
        successful_runs = [t for t in timing_summary if t['status'] == 'success']
        if successful_runs:
            print(f"\n{'='*60}")
            print("Performance Statistics")
            print(f"{'='*60}")
            avg_wncrod_time = np.mean([t['wncrod_time'] for t in successful_runs])
            avg_total_time = np.mean([t['total_time'] for t in successful_runs])
            total_wncrod_time = np.sum([t['wncrod_time'] for t in successful_runs])
            
            print(f"Average WNCROD computation time: {avg_wncrod_time:.2f} seconds ({avg_wncrod_time/60:.2f} minutes)")
            print(f"Average total time per dataset: {avg_total_time:.2f} seconds ({avg_total_time/60:.2f} minutes)")
            print(f"Total WNCROD computation time: {total_wncrod_time:.2f} seconds ({total_wncrod_time/60:.2f} minutes)")
        
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