#!/bin/bash

# Script to run OGP-NP continual learning experiments with varying epochs,
# fixed heads, specific seeds, and push results to Git.
# Assumes the Python script also includes epochs in the created results folder name.

# --- Configuration ---
PYTHON_SCRIPT_NAME="ogp_np_cl.py" # Make sure this matches your Python script filename
GIT_REMOTE_NAME="origin" # Change if your remote is named differently
GIT_BRANCH_NAME=$(git rev-parse --abbrev-ref HEAD) # Get current branch name

# Seeds to run (5 different seeds)
SEEDS=(0 1 2 3 4)

# Experiment types
EXPERIMENT_TYPES=("PermutedMNIST" "SplitMNIST" "RotatedMNIST")

# Epochs per task values to iterate over
EPOCHS_PER_TASK_VALUES=(10 20 50)

# --- Check if Python script exists ---
if [ ! -f "$PYTHON_SCRIPT_NAME" ]; then
    echo "Error: Python script '$PYTHON_SCRIPT_NAME' not found!"
    exit 1
fi

# --- Function to perform Git operations ---
perform_git_operations() {
    local results_dir_pattern="$1" # This pattern now includes epochs
    local commit_message="$2"

    local found_dir
    # Find the most recently created directory matching the full pattern (including epochs)
    # The Python script appends a timestamp, so we use '*' at the end.
    found_dir=$(find . -maxdepth 1 -type d -name "${results_dir_pattern}*" -printf '%T@ %p\n' | sort -nr | head -n1 | cut -d' ' -f2-)
    
    if [ -z "$found_dir" ] || [ ! -d "$found_dir" ]; then
        echo "Error: Could not find results directory matching pattern '$results_dir_pattern'."
        echo "Looked for: ${results_dir_pattern}*"
        echo "Skipping Git operations for this run."
        return
    fi
    
    RESULTS_DIR_TO_ADD=$(basename "$found_dir")

    echo "Found results directory: $RESULTS_DIR_TO_ADD"

    echo "Adding results to Git..."
    git add "$RESULTS_DIR_TO_ADD"
    
    echo "Committing results..."
    git commit -m "$commit_message"
    
    echo "Pushing to remote '$GIT_REMOTE_NAME' branch '$GIT_BRANCH_NAME'..."
    git push "$GIT_REMOTE_NAME" "$GIT_BRANCH_NAME"
    
    if [ $? -eq 0 ]; then
        echo "Git push successful."
    else
        echo "Error: Git push failed."
        # Consider adding 'exit 1' here if a failed push should stop all experiments
    fi
    sleep 2 # Small delay
}


# --- Main Experiment Loop ---
for EXP_TYPE in "${EXPERIMENT_TYPES[@]}"; do
    echo "================================================================="
    echo "Starting Experiment Type: $EXP_TYPE"
    echo "================================================================="

    NUM_CL_TASKS=0
    if [ "$EXP_TYPE" == "PermutedMNIST" ]; then
        NUM_CL_TASKS=10
    elif [ "$EXP_TYPE" == "SplitMNIST" ]; then
        NUM_CL_TASKS=5 # Based on fixed 2 classes per split in Python Config
    elif [ "$EXP_TYPE" == "RotatedMNIST" ]; then
        NUM_CL_TASKS=10
    else
        echo "Unknown experiment type: $EXP_TYPE. Skipping."
        continue
    fi
    echo "Derived number of tasks for $EXP_TYPE: $NUM_CL_TASKS"

    for EPOCHS_CURRENT_RUN in "${EPOCHS_PER_TASK_VALUES[@]}"; do
        echo "-----------------------------------------------------"
        echo "Running with Epochs Per Task: $EPOCHS_CURRENT_RUN"
        echo "-----------------------------------------------------"

        for SEED in "${SEEDS[@]}"; do
            echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
            echo "Running for Seed: $SEED"
            echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

            # --- Run: Fixed Head Per Task ---
            MODE_STR="FixedHead" # Only this mode now
            echo "Running $EXP_TYPE with $MODE_STR (Seed: $SEED, Tasks: $NUM_CL_TASKS, Epochs: $EPOCHS_CURRENT_RUN)"
            
            python "$PYTHON_SCRIPT_NAME" \
                --experiment_type "$EXP_TYPE" \
                --num_tasks "$NUM_CL_TASKS" \
                --epochs_per_task "$EPOCHS_CURRENT_RUN" \
                --seed "$SEED" \
                --fixed_head_per_task 
            
            echo "$MODE_STR run finished for Seed $SEED, Epochs $EPOCHS_CURRENT_RUN."
            
            COMMIT_MSG="Results: $EXP_TYPE $MODE_STR, $NUM_CL_TASKS tasks, $EPOCHS_CURRENT_RUN EPT, Seed $SEED"
            
            # *** MODIFIED RESULTS_PATTERN to include epochs ***
            RESULTS_PATTERN="results_OGP-NP_${EXP_TYPE}_${MODE_STR}_${NUM_CL_TASKS}tasks_seed${SEED}_${EPOCHS_CURRENT_RUN}epochs"
            
            perform_git_operations "$RESULTS_PATTERN" "$COMMIT_MSG"
            echo "----------------------------------------"
        done # End SEED loop
    done # End EPOCHS_CURRENT_RUN loop
    echo "Finished all runs for Experiment Type: $EXP_TYPE"
done # End EXP_TYPE loop

echo "================================================================="
echo "All experiments completed."
echo "================================================================="
