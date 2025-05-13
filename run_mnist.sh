#!/bin/bash

# --- Configuration ---
PYTHON_SCRIPT_NAME="ogp_mnist.py" # Name of your Python script
DEFAULT_NUM_TASKS=10                      # Number of tasks for Permuted/Rotated MNIST

# --- Git Configuration ---
GIT_PUSH_ENABLED=true                     # Set to true to enable pushing to GitHub, false to disable
GIT_REMOTE_NAME="origin"                  # Your Git remote name
GIT_BRANCH_NAME="main"                    # The branch to push to

# --- Experiment Parameters ---
EXPERIMENT_TYPES=("PermutedMNIST" "RotatedMNIST")
SEEDS=(100 101 102)
EPOCHS_PER_TASK_OPTIONS=(10 50)
THRESHOLDS_FOR_DYNAMIC=(0.01 0.005 0.001) # For dynamic head allocation

# --- Helper Function to Construct Results Directory Name ---
# This function should mirror the logic in your Python script's __main__ block
construct_results_dir_name() {
    local exp_type=$1
    local seed_val=$2
    local epochs_val=$3
    local fixed_head_mode=$4 # "FixedHead" or "DynamicHead"
    local threshold_val=$5   # Only used if fixed_head_mode is "DynamicHead"

    if [ "$fixed_head_mode" == "FixedHead" ]; then
        echo "${exp_type}_seed${seed_val}_FixedHead_${epochs_val}epochs"
    else
        # Ensure threshold_val is passed correctly for dynamic mode
        echo "${exp_type}_seed${seed_val}_DynamicHead_${epochs_val}epochs_thresh${threshold_val}"
    fi
}

# --- Helper Function to Push Results to GitHub ---
push_results_to_github() {
    local results_dir=$1
    local commit_message=$2

    if [ "$GIT_PUSH_ENABLED" != "true" ]; then
        echo "      Git push is disabled. Skipping."
        return
    fi

    if [ ! -d ".git" ]; then
        echo "      WARNING: Not a Git repository. Cannot push results."
        return
    fi

    if [ ! -d "$results_dir" ]; then
        echo "      WARNING: Results directory '$results_dir' not found. Cannot push."
        return
    fi

    echo "      Attempting to push results for: $results_dir"
    
    git add "$results_dir"
    if git commit -m "$commit_message"; then
        echo "      Successfully committed: $commit_message"
        if git push "$GIT_REMOTE_NAME" "$GIT_BRANCH_NAME"; then
            echo "      Successfully pushed to $GIT_REMOTE_NAME $GIT_BRANCH_NAME."
        else
            echo "      ERROR: Failed to push to GitHub. Please check your Git remote and authentication."
        fi
    else
        echo "      WARNING: Git commit failed. Nothing to commit or other error for '$results_dir'."
        echo "      This might happen if the results directory was empty or unchanged."
    fi
}


# --- Main Experiment Loop ---
echo "Starting OGP-NP Continual Learning Experiments for MNIST Variants..."
echo "Python script: $PYTHON_SCRIPT_NAME"
echo "Git Push Enabled: $GIT_PUSH_ENABLED (Remote: $GIT_REMOTE_NAME, Branch: $GIT_BRANCH_NAME)"
echo "==================================================================="

# Loop through each experiment type
for EXP_TYPE in "${EXPERIMENT_TYPES[@]}"; do
    echo ""
    echo "----------------------------------------------------"
    echo "EXPERIMENT TYPE: $EXP_TYPE"
    echo "----------------------------------------------------"

    # Loop through each seed
    for SEED_VAL in "${SEEDS[@]}"; do
        echo ""
        echo "  SEED: $SEED_VAL"
        echo "  ----------------------------------"

        # Loop through each epoch option
        for EPOCHS in "${EPOCHS_PER_TASK_OPTIONS[@]}"; do
            echo ""
            echo "    EPOCHS PER TASK: $EPOCHS"
            echo "    .............................."

            # Strategy 1: Fixed head per task
            echo ""
            echo "      Strategy: Fixed head per task"
            CMD="python \"$PYTHON_SCRIPT_NAME\" \
                --experiment_type \"$EXP_TYPE\" \
                --num_tasks \"$DEFAULT_NUM_TASKS\" \
                --epochs_per_task \"$EPOCHS\" \
                --seed \"$SEED_VAL\" \
                --fixed_head_per_task"
            echo "      Running: $CMD"
            eval "$CMD" 
            
            RESULTS_DIR_FIXED=$(construct_results_dir_name "$EXP_TYPE" "$SEED_VAL" "$EPOCHS" "FixedHead")
            COMMIT_MSG_FIXED="Results: $EXP_TYPE, Seed $SEED_VAL, Epochs $EPOCHS, Fixed Heads"
            push_results_to_github "$RESULTS_DIR_FIXED" "$COMMIT_MSG_FIXED"
            
            echo "      Finished fixed head strategy for $EXP_TYPE, Seed $SEED_VAL, Epochs $EPOCHS."
            echo "      .............................."


            # Strategy 2: Dynamic head allocation with different thresholds
            for THRESHOLD_VAL in "${THRESHOLDS_FOR_DYNAMIC[@]}"; do
                echo ""
                echo "      Strategy: Dynamic head allocation"
                echo "      THRESHOLD: $THRESHOLD_VAL"
                CMD="python \"$PYTHON_SCRIPT_NAME\" \
                    --experiment_type \"$EXP_TYPE\" \
                    --num_tasks \"$DEFAULT_NUM_TASKS\" \
                    --epochs_per_task \"$EPOCHS\" \
                    --seed \"$SEED_VAL\" \
                    --threshold \"$THRESHOLD_VAL\""
                echo "      Running: $CMD"
                eval "$CMD"

                RESULTS_DIR_DYNAMIC=$(construct_results_dir_name "$EXP_TYPE" "$SEED_VAL" "$EPOCHS" "DynamicHead" "$THRESHOLD_VAL")
                COMMIT_MSG_DYNAMIC="Results: $EXP_TYPE, Seed $SEED_VAL, Epochs $EPOCHS, Dynamic Heads (Thresh: $THRESHOLD_VAL)"
                push_results_to_github "$RESULTS_DIR_DYNAMIC" "$COMMIT_MSG_DYNAMIC"

                echo "      Finished dynamic head (threshold $THRESHOLD_VAL) for $EXP_TYPE, Seed $SEED_VAL, Epochs $EPOCHS."
                echo "      .............................."
            done # End of thresholds loop
        done # End of epochs loop
    done # End of seeds loop
done # End of experiment types loop

echo ""
echo "==================================================================="
echo "All specified MNIST experiments completed."
echo "==================================================================="
