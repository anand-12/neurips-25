#!/bin/bash

# Script to run OGP-NP continual learning experiments

# --- Configuration ---
PYTHON_SCRIPT_NAME="save_res.py" # Make sure this matches your Python script filename
BASE_EPOCHS_PER_TASK=5

# Seeds to run
SEEDS=(11 22 33 44 55 66 77 88 99 1010) # 10 different seeds

# Experiment types
EXPERIMENT_TYPES=("PermutedMNIST" "SplitMNIST" "RotatedMNIST")

# Z-Divergence Thresholds for dynamic head allocation
# For FixedHead mode, this list is not used, but we run it once with --fixed_head_per_task
DIVERGENCE_THRESHOLDS=(0.1 0.2 0.5 1.0)

# --- Check if Python script exists ---
if [ ! -f "$PYTHON_SCRIPT_NAME" ]; then
    echo "Error: Python script '$PYTHON_SCRIPT_NAME' not found!"
    exit 1
fi

# --- Main Experiment Loop ---
for EXP_TYPE in "${EXPERIMENT_TYPES[@]}"; do
    echo "================================================================="
    echo "Starting Experiment Type: $EXP_TYPE"
    echo "================================================================="

    NUM_CL_TASKS=0
    # Determine number of tasks based on experiment type
    # (Matches logic in your Python script's prepare_task_data and Config)
    if [ "$EXP_TYPE" == "PermutedMNIST" ]; then
        NUM_CL_TASKS=10 # As per your request
    elif [ "$EXP_TYPE" == "SplitMNIST" ]; then
        # classes_per_split is fixed to 2 in your Python Config
        # So, 10 classes / 2 classes_per_split = 5 tasks
        NUM_CL_TASKS=5
    elif [ "$EXP_TYPE" == "RotatedMNIST" ]; then
        NUM_CL_TASKS=10 # As per your request
    else
        echo "Unknown experiment type: $EXP_TYPE. Skipping."
        continue
    fi

    echo "Derived number of tasks for $EXP_TYPE: $NUM_CL_TASKS"

    for SEED in "${SEEDS[@]}"; do
        echo "-----------------------------------------------------"
        echo "Running for Seed: $SEED"
        echo "-----------------------------------------------------"

        # --- Run 1: Fixed Head Per Task ---
        echo "Running $EXP_TYPE with Fixed Head Per Task (Seed: $SEED, Tasks: $NUM_CL_TASKS, Epochs: $BASE_EPOCHS_PER_TASK)"
        python "$PYTHON_SCRIPT_NAME" \
            --experiment_type "$EXP_TYPE" \
            --num_tasks "$NUM_CL_TASKS" \
            --epochs_per_task "$BASE_EPOCHS_PER_TASK" \
            --seed "$SEED" \
            --fixed_head_per_task
        
        echo "Fixed Head Per Task run finished for Seed $SEED."
        echo "----------------------------------------"


        # --- Run 2 onwards: Dynamic Head Allocation with different thresholds ---
        for Z_THRESH in "${DIVERGENCE_THRESHOLDS[@]}"; do
            echo "Running $EXP_TYPE with Dynamic Heads (Seed: $SEED, Tasks: $NUM_CL_TASKS, Epochs: $BASE_EPOCHS_PER_TASK, Z-Div-Thresh: $Z_THRESH)"
            # Note: --fixed_head_per_task is NOT passed here, so it defaults to False (dynamic)
            python "$PYTHON_SCRIPT_NAME" \
                --experiment_type "$EXP_TYPE" \
                --num_tasks "$NUM_CL_TASKS" \
                --epochs_per_task "$BASE_EPOCHS_PER_TASK" \
                --seed "$SEED" \
                --z_divergence_threshold "$Z_THRESH"
            
            echo "Dynamic Head (Z-Div-Thresh: $Z_THRESH) run finished for Seed $SEED."
            echo "----------------------------------------"
        done
    done
    echo "Finished all runs for Experiment Type: $EXP_TYPE"
done

echo "================================================================="
echo "All experiments completed."
echo "================================================================="
