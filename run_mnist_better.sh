#!/bin/bash

# --- Configuration ---
PYTHON_SCRIPT="og_npcl_static_head.py" # Make sure this is the correct name of your Python script
NUM_TASKS=10                         # Default number of tasks, adjust if needed
EPOCHS_PER_TASK=5                    # Default epochs per task, adjust if needed
SEED=42                              # Default random seed

# Experiment types to iterate over
EXPERIMENT_TYPES=("PermutedMNIST" "SplitMNIST" "RotatedMNIST")


THRESHOLDS_FOR_DYNAMIC=(0.01 0.005 0.001)

# --- Main Loop ---

echo "Starting OGP-NP Continual Learning Experiments..."
echo "Python script: $PYTHON_SCRIPT"
echo "===================================================="

# Loop through each experiment type (PermutedMNIST, SplitMNIST, RotatedMNIST)
for EXP_TYPE in "${EXPERIMENT_TYPES[@]}"; do
    echo ""
    echo "----------------------------------------------------"
    echo "EXPERIMENT TYPE: $EXP_TYPE"
    echo "----------------------------------------------------"

    # # Strategy 1: New head each time (fixed head per task)
    # echo ""
    # echo "  Strategy: Fixed head per task (new head each time)"
    # echo "  Running: python $PYTHON_SCRIPT --experiment_type \"$EXP_TYPE\" --num_tasks \"$NUM_TASKS\" --epochs_per_task \"$EPOCHS_PER_TASK\" --seed \"$SEED\" --fixed_head_per_task"
    
    # python "$PYTHON_SCRIPT" \
    #     --experiment_type "$EXP_TYPE" \
    #     --num_tasks "$NUM_TASKS" \
    #     --epochs_per_task "$EPOCHS_PER_TASK" \
    #     --seed "$SEED" \
    #     --fixed_head_per_task
    
    # echo "  Finished: Fixed head strategy for $EXP_TYPE."
    # echo ""
    # echo "  ----------------------------------"

    # Strategy 2: Dynamic head allocation with different Z_DIVERGENCE_THRESHOLD values
    for THRESHOLD in "${THRESHOLDS_FOR_DYNAMIC[@]}"; do
        echo ""
        echo "  Strategy: Dynamic head allocation"
        echo "  Z_DIVERGENCE_THRESHOLD: $THRESHOLD"
        echo "  (A new head is spawned if min_symKL >= $THRESHOLD)"
        echo "  Running: python $PYTHON_SCRIPT --experiment_type \"$EXP_TYPE\" --num_tasks \"$NUM_TASKS\" --epochs_per_task \"$EPOCHS_PER_TASK\" --seed \"$SEED\" --z_divergence_threshold \"$THRESHOLD\""

        python "$PYTHON_SCRIPT" \
            --experiment_type "$EXP_TYPE" \
            --num_tasks "$NUM_TASKS" \
            --epochs_per_task "$EPOCHS_PER_TASK" \
            --seed "$SEED" \
            --z_divergence_threshold "$THRESHOLD"
            # Note: --fixed_head_per_task is NOT included here for dynamic allocation

        echo "  Finished: Dynamic head (threshold $THRESHOLD) for $EXP_TYPE."
        echo ""
        echo "  ----------------------------------"
    done
done

echo "===================================================="
echo "All experiments completed."
echo "===================================================="
