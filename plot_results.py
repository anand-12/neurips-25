import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import glob
import argparse
from collections import defaultdict

def load_results_from_folder(folder_path):
    """Loads results.pt and config.json from a given folder."""
    results_pt_path = os.path.join(folder_path, 'results.pt')
    config_json_path = os.path.join(folder_path, 'config.json')

    if not os.path.exists(results_pt_path):
        print(f"Warning: results.pt not found in {folder_path}")
        return None
    if not os.path.exists(config_json_path):
        print(f"Warning: config.json not found in {folder_path}")
        return None

    try:
        try:
            results_data = torch.load(results_pt_path, map_location=torch.device('cpu'), weights_only=False)
        except RuntimeError as e:
            if "weights_only" in str(e): 
                 print(f"Info: Retrying torch.load with weights_only=True for {results_pt_path}")
                 results_data = torch.load(results_pt_path, map_location=torch.device('cpu'), weights_only=True)
            else:
                raise 
        with open(config_json_path, 'r') as f:
            config_data = json.load(f)
        return {"results": results_data, "config": config_data, "folder_path": folder_path}
    except Exception as e:
        print(f"Error loading data from {folder_path}: {e}")
        return None

def collect_all_experiment_data(root_dir="."):
    """Collects data from all relevant experiment folders."""
    all_data_points = []
    search_pattern = os.path.join(root_dir, "results_OGP-NP_*")
    experiment_folders = glob.glob(search_pattern)
    print(f"Found {len(experiment_folders)} potential experiment folders matching '{search_pattern}'.")

    for folder in experiment_folders:
        if os.path.isdir(folder):
            data_point = load_results_from_folder(folder)
            if data_point:
                all_data_points.append(data_point)
    
    print(f"Successfully loaded data from {len(all_data_points)} experiment runs.")
    return all_data_points

def extract_and_format_data(raw_data_points):
    """Extracts key info, focusing on final accuracies and EPOCHS_PER_TASK config."""
    formatted_data_list = []
    for data_point in raw_data_points:
        try:
            config = data_point['config']
            results = data_point['results']
            folder_path = data_point.get('folder_path', 'N/A')

            exp_type = config.get('EXPERIMENT_TYPE')
            fixed_head = config.get('FIXED_HEAD_PER_TASK', False)
            z_div_thresh_val = config.get('Z_DIVERGENCE_THRESHOLD') # Keep original value or None
            if not fixed_head and z_div_thresh_val is None:
                z_div_thresh_display = "N/A_Dynamic" # For display if dynamic but no threshold
            elif z_div_thresh_val is not None:
                z_div_thresh_display = str(z_div_thresh_val)
            else: # Fixed head, threshold not applicable for mode name
                z_div_thresh_display = ""


            seed = config.get('SEED')
            num_tasks_val = config.get('NUM_CL_TASKS_EFFECTIVE')
            num_tasks_effective = int(num_tasks_val) if num_tasks_val is not None and str(num_tasks_val).lower() != 'none' else None

            avg_acc_over_time = results.get('average_accuracies_over_time')
            
            epochs_per_task_config_val = config.get('EPOCHS_PER_TASK')
            epochs_per_task_config = int(epochs_per_task_config_val) if epochs_per_task_config_val is not None else None

            valid_data = (
                exp_type and 
                avg_acc_over_time is not None and 
                seed is not None and 
                num_tasks_effective is not None and
                epochs_per_task_config is not None
            )
            if not valid_data:
                print(f"Warning: Missing essential data in {folder_path}. Skipping.")
                continue
            
            if len(avg_acc_over_time) != num_tasks_effective:
                print(f"Warning: Mismatch in avg_acc_over_time length ({len(avg_acc_over_time)}) and num_tasks ({num_tasks_effective}) for {folder_path}. Skipping.")
                continue
            
            head_mode = "FixedHead" if fixed_head else f"DynamicSKL_Z{z_div_thresh_display}"
            
            formatted_data_list.append({
                "experiment_type": exp_type,
                "head_mode": head_mode,
                "seed": seed,
                "num_tasks_effective": num_tasks_effective,
                "average_accuracies_over_time": np.array(avg_acc_over_time),
                "epochs_per_task_config": epochs_per_task_config
            })

        except Exception as e:
            print(f"Error processing data for folder {data_point.get('folder_path', 'N/A')}: {e}")
    
    print(f"Successfully formatted {len(formatted_data_list)} experiment runs.")
    return formatted_data_list


def aggregate_results_for_single_plot(formatted_data_list):
    """
    Aggregates results by (experiment_type, head_mode, epochs_per_task_config) across seeds.
    Calculates median and percentiles for 'average_accuracies_over_time'.
    """
    aggregated_temp = defaultdict(lambda: {
        'final_accuracies_all_seeds': [], 
        'num_tasks_list': []
    })

    for data_point in formatted_data_list:
        key = (
            data_point['experiment_type'],
            data_point['head_mode'],
            data_point['epochs_per_task_config'] # Include epochs config in the key
        )
        aggregated_temp[key]['final_accuracies_all_seeds'].append(data_point['average_accuracies_over_time'])
        aggregated_temp[key]['num_tasks_list'].append(data_point['num_tasks_effective'])

    final_aggregated = {} # Key: (exp_type, head_mode, epochs_config) -> stats

    for key_tuple, data in aggregated_temp.items():
        # exp_type, head_mode, epochs_config = key_tuple (already unpacked by key_tuple)

        if not data['final_accuracies_all_seeds'] or not data['num_tasks_list']:
            print(f"Warning: No final accuracy data or num_tasks for {key_tuple}. Skipping.")
            continue

        first_num_tasks = data['num_tasks_list'][0]
        if not all(nt == first_num_tasks for nt in data['num_tasks_list']):
            print(f"Warning: Inconsistent num_tasks for {key_tuple}. Skipping.")
            continue
        
        valid_final_accuracies = [acc for acc in data['final_accuracies_all_seeds'] if len(acc) == first_num_tasks]
        if not valid_final_accuracies:
            print(f"Warning: No valid final accuracy lists for {key_tuple} after length check. Skipping.")
            continue
        
        stacked_final_acc = np.stack(valid_final_accuracies) # [seeds, tasks]
        median_final_acc = np.median(stacked_final_acc, axis=0)
        p25_final_acc = np.percentile(stacked_final_acc, 25, axis=0)
        p75_final_acc = np.percentile(stacked_final_acc, 75, axis=0)
        num_seeds_final = len(valid_final_accuracies)

        final_aggregated[key_tuple] = {
            'median_final': median_final_acc,
            'p25_final': p25_final_acc,
            'p75_final': p75_final_acc,
            'num_tasks': first_num_tasks,
            'num_seeds': num_seeds_final
        }
            
    return final_aggregated


def process_single_file_for_single_plot(file_path):
    """Loads a single results.pt and its config, formats for the single plot structure."""
    folder_path = os.path.dirname(file_path)
    data_point = load_results_from_folder(folder_path)
    if not data_point:
        return None

    formatted_list = extract_and_format_data([data_point]) 
    if not formatted_list:
        return None
    
    single_run_data = formatted_list[0]
    
    key = (
        single_run_data['experiment_type'],
        single_run_data['head_mode'],
        single_run_data['epochs_per_task_config']
    )
    final_acc = single_run_data['average_accuracies_over_time']

    single_file_aggregated = {
        key: {
            'median_final': final_acc,
            'p25_final': final_acc, 
            'p75_final': final_acc,
            'num_tasks': single_run_data['num_tasks_effective'],
            'num_seeds': 1
        }
    }
    return single_file_aggregated


def plot_all_in_one(aggregated_data, output_dir=".", single_file_mode=False):
    """
    Generates a single plot containing all experiment combinations.
    Lines are (experiment_type, head_mode, epochs_per_task_config).
    """
    if not aggregated_data:
        print("No data provided to plot_all_in_one.")
        return

    plt.figure(figsize=(15, 9)) # Adjusted figure size for potentially many lines
    plot_title_suffix = " (Single Run)" if single_file_mode else " (Aggregated)"
    
    max_num_tasks_for_plot = 0

    # Sort keys for consistent plotting order: exp_type, then epochs_config, then head_mode
    # Custom sort for head_mode: FixedHead first, then DynamicSKL_Z by Z value
    def sort_key_func(item):
        key_tuple, _ = item
        exp_type, head_mode, epochs_config = key_tuple
        
        # For head_mode sorting
        is_fixed = (head_mode == "FixedHead")
        dynamic_z_val = float('inf')
        if "DynamicSKL_Z" in head_mode:
            try:
                # Extract numeric part after 'Z'
                z_str = head_mode.split('Z')[-1]
                if z_str not in ["N/A", "N/A_Dynamic"]:
                    dynamic_z_val = float(z_str)
            except ValueError: # Handle cases like "N/A_Dynamic"
                pass # Keep inf
        
        return (exp_type, epochs_config, not is_fixed, dynamic_z_val)


    sorted_items = sorted(aggregated_data.items(), key=sort_key_func)

    for key_tuple, data in sorted_items:
        exp_type, head_mode, epochs_config = key_tuple
        
        if not data or data.get('median_final') is None:
            continue

        num_tasks = data['num_tasks']
        max_num_tasks_for_plot = max(max_num_tasks_for_plot, num_tasks)
        median_acc = data['median_final']
        p25_acc = data['p25_final']
        p75_acc = data['p75_final']
        
        x_axis = np.arange(1, num_tasks + 1)
        
        label = f"{exp_type} - {head_mode} - {epochs_config} Epc"
        if not single_file_mode:
            label += f" (N_seeds={data['num_seeds']})"

        plt.plot(x_axis, median_acc, marker='o', linestyle='-', label=label, markersize=5) # Smaller markers
        if data['num_seeds'] > 1 or single_file_mode: 
            plt.fill_between(x_axis, p25_acc, p75_acc, alpha=0.15) # Slightly less alpha
    
    if max_num_tasks_for_plot > 0:
        plt.xlabel("Number of Tasks Trained")
        plt.ylabel("Average Accuracy (%) over Seen Tasks (Median)")
        plt.title(f"OGP-NP Continual Learning Performance Comparison{plot_title_suffix}")
        plt.xticks(np.arange(1, max_num_tasks_for_plot + 1))
        current_y_min = plt.gca().get_ylim()[0]
        plt.ylim(min(75, current_y_min if current_y_min < 100 else 75) , 101) 
        plt.legend(loc='best', fontsize='x-small', ncol=1) # Adjust ncol based on number of lines
        plt.grid(True, linestyle='--', alpha=0.7)
        
        file_suffix = "_single_all_in_one" if single_file_mode else "_aggregated_all_in_one"
        plot_filename = os.path.join(output_dir, f"ACC_vs_Tasks_ALL{file_suffix}.png")
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight') # Add bbox_inches
        print(f"Plot saved: {plot_filename}")
    else:
        print(f"No valid data found to generate the combined plot.")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot results from OGP-NP experiments, all in one plot.")
    parser.add_argument('--results_root_dir', type=str, default=".",
                        help='Root directory containing the "results_OGP-NP_*" folders (for aggregation).')
    parser.add_argument('--output_plot_dir', type=str, default=".",
                        help='Directory to save the generated plots.')
    parser.add_argument('--specific_file_path', type=str, default=None,
                        help='Path to a specific results.pt file to plot individually. Overrides directory scan.')
    
    args = parser.parse_args()

    if not os.path.isdir(args.output_plot_dir):
        os.makedirs(args.output_plot_dir, exist_ok=True)
        print(f"Created output directory for plots: {args.output_plot_dir}")

    data_to_plot = None
    is_single_file_plot = False

    if args.specific_file_path:
        print(f"Processing single file: {args.specific_file_path}")
        if not os.path.exists(args.specific_file_path):
            print(f"Error: Specific file not found: {args.specific_file_path}")
            return
        if not args.specific_file_path.endswith("results.pt"):
            print(f"Error: --specific_file_path must point to a 'results.pt' file.")
            return
        
        config_path_check = os.path.join(os.path.dirname(args.specific_file_path), "config.json")
        if not os.path.exists(config_path_check):
            print(f"Error: Corresponding config.json not found at {config_path_check}")
            return

        data_to_plot = process_single_file_for_single_plot(args.specific_file_path)
        is_single_file_plot = True
    else:
        print(f"Scanning directory for multiple experiment results: {args.results_root_dir}")
        raw_collected_data = collect_all_experiment_data(args.results_root_dir)
        if not raw_collected_data:
            print("No data found from directory scan. Exiting.")
            return
        formatted_data = extract_and_format_data(raw_collected_data)
        if not formatted_data:
            print("No data successfully formatted. Exiting.")
            return
        data_to_plot = aggregate_results_for_single_plot(formatted_data)
        is_single_file_plot = False

    if not data_to_plot:
        print("No data available to plot after processing. Exiting.")
        return
        
    print("\nPlotting data...")
    plot_all_in_one(data_to_plot, args.output_plot_dir, single_file_mode=is_single_file_plot)
    
    print("\nProcessing complete.")

if __name__ == "__main__":
    main()
