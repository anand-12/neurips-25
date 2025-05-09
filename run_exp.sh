#!/bin/bash
# Script to run Neural Process Continual Learning experiments

# Create directories
mkdir -p results models data

# Run all experiments
echo "=== Running experiments ==="

# Run individual experiments (comment out what you don't need)

# Permuted MNIST experiment
echo "Running Permuted MNIST experiment..."
python main.py --experiment permuted_mnist 

# Split MNIST experiment
echo "Running Split MNIST experiment..."
python main.py --experiment split_mnist 

# Split CIFAR experiment
echo "Running Split CIFAR experiment..."
python main.py --experiment split_cifar 

# Run all experiments at once (uncomment if needed)
# python main.py --experiment all 

echo "=== All experiments completed! ==="
echo "Results saved in the 'results' directory."
echo "Models saved in the 'models' directory."
