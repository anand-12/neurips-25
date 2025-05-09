#!/bin/bash

# Grid Search for Neural Process Continual Learning
# This script runs the neural process model over a grid of beta and reg_weight values

# Create a results directory
mkdir -p results
mkdir -p outputs

# Define parameter grids
BETAS=(0.0001 0.001 0.01 0.1 0.5 1.0)
REG_WEIGHTS=(0.01 0.1 0.5 1.0)
SEEDS=(0 1 2)
EPOCHS=3

echo "Starting grid search with parameters:"
echo "Betas: ${BETAS[@]}"
echo "Regularization weights: ${REG_WEIGHTS[@]}"
echo "Seeds: ${SEEDS[@]}"
echo "Epochs: $EPOCHS"

# Create a CSV file to track all results
RESULTS_CSV="results/summary_results.csv"
echo "beta,reg_weight,seed,avg_forgetting,avg_final_accuracy" > $RESULTS_CSV

# Track total runs and progress
TOTAL_RUNS=$((${#BETAS[@]} * ${#REG_WEIGHTS[@]} * ${#SEEDS[@]}))
CURRENT_RUN=0

# Run the grid search
for BETA in "${BETAS[@]}"; do
    for REG_WEIGHT in "${REG_WEIGHTS[@]}"; do
        for SEED in "${SEEDS[@]}"; do
            CURRENT_RUN=$((CURRENT_RUN + 1))
            
            # Print progress
            echo "===================================================================================="
            echo "Run $CURRENT_RUN/$TOTAL_RUNS: beta=$BETA, reg_weight=$REG_WEIGHT, seed=$SEED"
            echo "===================================================================================="
            
            # Run the Python script with the current parameters
            python neural_process_cl.py --beta $BETA --reg_weight $REG_WEIGHT --seed $SEED --epochs $EPOCHS --output_prefix "run"
            
            # Extract results from the JSON file
            JSON_FILE="outputs/run_beta${BETA}_reg${REG_WEIGHT}_seed${SEED}.json"
            
            if [ -f "$JSON_FILE" ]; then
                # Extract the metrics using jq (if available) or python
                if command -v jq &> /dev/null; then
                    AVG_FORGETTING=$(jq '.avg_forgetting' "$JSON_FILE")
                    AVG_ACCURACY=$(jq '.avg_final_accuracy' "$JSON_FILE")
                else
                    # Fallback to Python for JSON parsing
                    AVG_FORGETTING=$(python -c "import json; print(json.load(open('$JSON_FILE'))['avg_forgetting'])")
                    AVG_ACCURACY=$(python -c "import json; print(json.load(open('$JSON_FILE'))['avg_final_accuracy'])")
                fi
                
                # Append to CSV
                echo "$BETA,$REG_WEIGHT,$SEED,$AVG_FORGETTING,$AVG_ACCURACY" >> $RESULTS_CSV
                echo "Results recorded: avg_forgetting=$AVG_FORGETTING, avg_accuracy=$AVG_ACCURACY"
            else
                echo "WARNING: Results file not found: $JSON_FILE"
            fi
        done
    done
done

echo "Grid search complete. Results saved to $RESULTS_CSV"

# Create summary analysis script
ANALYSIS_SCRIPT="results/analyze_results.py"
cat > $ANALYSIS_SCRIPT << 'EOF'
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Load results
df = pd.read_csv('results/summary_results.csv')

# Convert columns to numeric
df['beta'] = pd.to_numeric(df['beta'])
df['reg_weight'] = pd.to_numeric(df['reg_weight'])
df['avg_forgetting'] = pd.to_numeric(df['avg_forgetting'])
df['avg_final_accuracy'] = pd.to_numeric(df['avg_final_accuracy'])

# Group by beta and reg_weight, computing mean and std across seeds
grouped = df.groupby(['beta', 'reg_weight']).agg({
    'avg_forgetting': ['mean', 'std'],
    'avg_final_accuracy': ['mean', 'std']
}).reset_index()

# Rename columns for clarity
grouped.columns = ['beta', 'reg_weight', 'forgetting_mean', 'forgetting_std', 
                  'accuracy_mean', 'accuracy_std']

print("Results averaged across seeds:")
print(grouped)

# Save grouped results to CSV
grouped.to_csv('results/grouped_results.csv', index=False)

# Create heatmap for forgetting
plt.figure(figsize=(10, 8))
pivot_forget = grouped.pivot_table(index='beta', columns='reg_weight', values='forgetting_mean')
sns.heatmap(pivot_forget, annot=True, fmt=".3f", cmap="YlOrRd", 
            xticklabels=pivot_forget.columns, yticklabels=pivot_forget.index)
plt.title('Average Forgetting by Beta and Regularization Weight')
plt.xlabel('Regularization Weight (λ)')
plt.ylabel('Beta (β)')
plt.savefig('results/forgetting_heatmap.png')

# Create heatmap for accuracy
plt.figure(figsize=(10, 8))
pivot_acc = grouped.pivot_table(index='beta', columns='reg_weight', values='accuracy_mean')
sns.heatmap(pivot_acc, annot=True, fmt=".3f", cmap="viridis", 
            xticklabels=pivot_acc.columns, yticklabels=pivot_acc.index)
plt.title('Average Final Accuracy by Beta and Regularization Weight')
plt.xlabel('Regularization Weight (λ)')
plt.ylabel('Beta (β)')
plt.savefig('results/accuracy_heatmap.png')

# Plot the relationship between beta and forgetting for different reg_weights
plt.figure(figsize=(10, 6))
for reg in sorted(grouped['reg_weight'].unique()):
    data = grouped[grouped['reg_weight'] == reg]
    plt.errorbar(data['beta'], data['forgetting_mean'], yerr=data['forgetting_std'], 
                 label=f'λ={reg}', marker='o')
plt.xscale('log')
plt.xlabel('Beta (β) - log scale')
plt.ylabel('Average Forgetting')
plt.title('Effect of Beta on Forgetting for Different Regularization Weights')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('results/beta_vs_forgetting.png')

# Plot the relationship between reg_weight and forgetting for different betas
plt.figure(figsize=(10, 6))
for beta in sorted(grouped['beta'].unique()):
    data = grouped[grouped['beta'] == beta]
    plt.errorbar(data['reg_weight'], data['forgetting_mean'], yerr=data['forgetting_std'], 
                 label=f'β={beta}', marker='o')
plt.xscale('log')
plt.xlabel('Regularization Weight (λ) - log scale')
plt.ylabel('Average Forgetting')
plt.title('Effect of Regularization Weight on Forgetting for Different Beta Values')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('results/reg_vs_forgetting.png')

# Find best parameters
best_accuracy = grouped.loc[grouped['accuracy_mean'].idxmax()]
best_forgetting = grouped.loc[grouped['forgetting_mean'].idxmin()]

print("\nParameters with highest average accuracy:")
print(f"Beta: {best_accuracy['beta']}, Reg Weight: {best_accuracy['reg_weight']}")
print(f"Accuracy: {best_accuracy['accuracy_mean']:.4f} ± {best_accuracy['accuracy_std']:.4f}")
print(f"Forgetting: {best_accuracy['forgetting_mean']:.4f} ± {best_accuracy['forgetting_std']:.4f}")

print("\nParameters with lowest forgetting:")
print(f"Beta: {best_forgetting['beta']}, Reg Weight: {best_forgetting['reg_weight']}")
print(f"Accuracy: {best_forgetting['accuracy_mean']:.4f} ± {best_forgetting['accuracy_std']:.4f}")
print(f"Forgetting: {best_forgetting['forgetting_mean']:.4f} ± {best_forgetting['forgetting_std']:.4f}")

# Find trade-off configurations (high accuracy and low forgetting)
# Normalize metrics to [0,1] range
grouped['norm_accuracy'] = (grouped['accuracy_mean'] - grouped['accuracy_mean'].min()) / (grouped['accuracy_mean'].max() - grouped['accuracy_mean'].min())
grouped['norm_forgetting'] = 1 - (grouped['forgetting_mean'] - grouped['forgetting_mean'].min()) / (grouped['forgetting_mean'].max() - grouped['forgetting_mean'].min())

# Calculate combined score (higher is better)
grouped['combined_score'] = 0.5 * grouped['norm_accuracy'] + 0.5 * grouped['norm_forgetting']

best_tradeoff = grouped.loc[grouped['combined_score'].idxmax()]
print("\nBest trade-off parameters (balanced accuracy and forgetting):")
print(f"Beta: {best_tradeoff['beta']}, Reg Weight: {best_tradeoff['reg_weight']}")
print(f"Accuracy: {best_tradeoff['accuracy_mean']:.4f} ± {best_tradeoff['accuracy_std']:.4f}")
print(f"Forgetting: {best_tradeoff['forgetting_mean']:.4f} ± {best_tradeoff['forgetting_std']:.4f}")
print(f"Combined Score: {best_tradeoff['combined_score']:.4f}")

# Save results summary
with open('results/summary.txt', 'w') as f:
    f.write("Grid Search Results Summary\n")
    f.write("==========================\n\n")
    
    f.write("Parameters with highest average accuracy:\n")
    f.write(f"Beta: {best_accuracy['beta']}, Reg Weight: {best_accuracy['reg_weight']}\n")
    f.write(f"Accuracy: {best_accuracy['accuracy_mean']:.4f} ± {best_accuracy['accuracy_std']:.4f}\n")
    f.write(f"Forgetting: {best_accuracy['forgetting_mean']:.4f} ± {best_accuracy['forgetting_std']:.4f}\n\n")
    
    f.write("Parameters with lowest forgetting:\n")
    f.write(f"Beta: {best_forgetting['beta']}, Reg Weight: {best_forgetting['reg_weight']}\n")
    f.write(f"Accuracy: {best_forgetting['accuracy_mean']:.4f} ± {best_forgetting['accuracy_std']:.4f}\n")
    f.write(f"Forgetting: {best_forgetting['forgetting_mean']:.4f} ± {best_forgetting['forgetting_std']:.4f}\n\n")
    
    f.write("Best trade-off parameters (balanced accuracy and forgetting):\n")
    f.write(f"Beta: {best_tradeoff['beta']}, Reg Weight: {best_tradeoff['reg_weight']}\n")
    f.write(f"Accuracy: {best_tradeoff['accuracy_mean']:.4f} ± {best_tradeoff['accuracy_std']:.4f}\n")
    f.write(f"Forgetting: {best_tradeoff['forgetting_mean']:.4f} ± {best_tradeoff['forgetting_std']:.4f}\n")
    f.write(f"Combined Score: {best_tradeoff['combined_score']:.4f}\n")

print("\nAnalysis complete. See results/summary.txt for key findings.")
EOF

echo "Analysis script created at $ANALYSIS_SCRIPT"
echo "Run 'python $ANALYSIS_SCRIPT' after grid search to analyze results"