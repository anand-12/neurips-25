import torch
import argparse
import numpy as np # For pretty printing numpy arrays if they exist
import json # For pretty printing config if it's a string

def display_results(data, current_key="root", indent=0): # Added current_key for context
    """
    Recursively displays the contents of the loaded dictionary.
    Handles nested dictionaries, lists, tensors, and numpy arrays.
    """
    prefix = "  " * indent
    if isinstance(data, dict):
        for key, value in data.items():
            print(f"{prefix}{key}:")
            display_results(value, key, indent + 1) # Pass key for context
    elif isinstance(data, list):
        if any(isinstance(item, (dict, list)) for item in data) and len(data) > 0 and len(data) < 10: # Expand short lists of complex items
             print(f"{prefix}[")
             for i, item in enumerate(data):
                 display_results(item, f"item_{i}", indent + 2)
             print(f"{prefix}]")
        else: 
            if len(data) > 10:
                # For lists of simple types or very long lists of complex types, summarize
                item_summary = type(data[0]).__name__ if data else "empty"
                print(f"{prefix}[ {len(data)} items of type ~{item_summary} ] (First 5: {data[:5]})")
            elif len(data) == 0:
                print(f"{prefix}[] (empty list)")
            else: # Short list of simple items
                print(f"{prefix}{data}")

    elif isinstance(data, torch.Tensor):
        print(f"{prefix}Tensor (shape: {data.shape}, dtype: {data.dtype}, device: {data.device})")
        if data.numel() < 20: 
            print(f"{prefix}  Values: {data.tolist()}")
        else:
            print(f"{prefix}  (Tensor data preview omitted for brevity)")
    elif isinstance(data, np.ndarray):
        print(f"{prefix}NumPy Array (shape: {data.shape}, dtype: {data.dtype})")
        if data.size < 20: 
            print(f"{prefix}  Values: {data.tolist()}")
        else:
            print(f"{prefix}  (NumPy array data preview omitted for brevity)")
    elif isinstance(data, str) and current_key == "config_used": # Check current_key
        try:
            config_dict = json.loads(data) # data is already the string value
            print(f"{prefix}Parsed Config:")
            display_results(config_dict, "config_details", indent +1) # Pass new key
        except json.JSONDecodeError:
            print(f"{prefix}Config String (raw):")
            for line in data.splitlines(): # data is the string
                print(f"{prefix}  {line}")
    else:
        print(f"{prefix}{data}")

def main():
    parser = argparse.ArgumentParser(description="Read and display contents of a PyTorch .pt results file.")
    parser.add_argument("filepath", type=str, help="Path to the .pt file")
    parser.add_argument("--map_location", type=str, default=None,
                        help="Map location for torch.load (e.g., 'cpu', 'cuda:0'). Defaults to loading on original device.")
    args = parser.parse_args()

    try:
        print(f"Attempting to load results from: {args.filepath}")
        map_location_arg = args.map_location if args.map_location else None
        
        # --- MODIFICATION: Set weights_only=False ---
        # This is necessary if the .pt file contains more than just model weights,
        # e.g., NumPy arrays, Python dicts/lists, or other pickled objects.
        # Only use weights_only=False if you trust the source of the .pt file.
        print("Loading with weights_only=False (assuming trusted source).")
        loaded_data = torch.load(args.filepath, map_location=map_location_arg, weights_only=False)
        # --- END MODIFICATION ---
        
        print("\n--- Contents of the .pt file ---")
        if isinstance(loaded_data, dict):
            display_results(loaded_data) # Initial call with default key "root"
        else:
            print("The loaded data is not a dictionary. Displaying raw data:")
            print(loaded_data)
        print("\n--- End of Contents ---")

    except FileNotFoundError:
        print(f"Error: File not found at {args.filepath}")
    except RuntimeError as e:
        print(f"A RuntimeError occurred: {e}")
        print("This might be related to loading tensors saved on a different device or an issue with unpickling.")
        if "weights_only" in str(e) and "numpy._core.multiarray.scalar" in str(e):
            print("Suggestion: The file contains NumPy scalars. Loading with `weights_only=False` (as attempted) is usually the fix if the file is trusted.")
            print("If the error persists with `weights_only=False`, the file might be corrupted or contain other unsupported types for the current environment.")
        elif "pickle" in str(e).lower():
             print("This looks like a pickling/unpickling error. Ensure the environment you are loading in has all the necessary class definitions if custom objects were saved.")
    except Exception as e:
        print(f"An unexpected error occurred while loading or displaying the file: {e}")

if __name__ == "__main__":
    main()
