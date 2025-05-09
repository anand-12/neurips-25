"""
Visualization utilities for continual learning experiments.
"""
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List


def plot_learning_curves(all_accuracies: Dict[int, List[float]], title: str, save_path: str = None):
    """
    Plot learning curves for all tasks across epochs.
    
    Args:
        all_accuracies: Dictionary mapping task_id to list of accuracies
        title: Title for the plot
        save_path: Path to save the plot (if None, only display)
    """
    plt.figure(figsize=(12, 6))
    
    for task_id, accuracies in all_accuracies.items():
        epochs = range(1, len(accuracies) + 1)
        plt.plot(epochs, accuracies, label=f'Task {task_id+1}')
    
    plt.xlabel('Epochs')
    plt.ylabel('Test Accuracy')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    
    plt.close()


def plot_task_matrix(task_task_accuracies: np.ndarray, task_labels: List[str] = None, title: str = None, save_path: str = None):
    """
    Plot the task-task accuracy matrix.
    
    Args:
        task_task_accuracies: 2D array where task_task_accuracies[i][j] is the accuracy on task j after training on task i
        task_labels: Labels for the tasks (if None, use Task 1, Task 2, etc.)
        title: Title for the plot
        save_path: Path to save the plot (if None, only display)
    """
    num_tasks = task_task_accuracies.shape[0]
    
    # Default task labels if not provided
    if task_labels is None:
        task_labels = [f"Task {i+1}" for i in range(num_tasks)]
    
    # Create figure
    plt.figure(figsize=(12, 10))
    plt.imshow(task_task_accuracies, cmap='magma', vmin=0, vmax=1)
    plt.colorbar(label='Accuracy')
    plt.xlabel('Evaluated on Task')
    plt.ylabel('After Training on Task')
    
    # Set title
    if title:
        plt.title(title)
    else:
        plt.title('Task-Task Accuracy Matrix')
    
    # Set axis labels
    plt.xticks(range(num_tasks), task_labels, rotation=45, ha="right")
    plt.yticks(range(num_tasks), task_labels)
    
    # Add accuracy values to cells
    for i in range(num_tasks):
        for j in range(num_tasks):
            if i >= j:  # Only show values for trained tasks
                color = 'white' if task_task_accuracies[i, j] < 0.7 else 'black'
                plt.text(j, i, f'{task_task_accuracies[i, j]:.2f}', 
                        ha='center', va='center', color=color)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    
    plt.close()


def plot_average_accuracies(task_task_accuracies: np.ndarray, title: str = None, save_path: str = None):
    """
    Plot the average accuracy after training on each task.
    
    Args:
        task_task_accuracies: 2D array where task_task_accuracies[i][j] is the accuracy on task j after training on task i
        title: Title for the plot
        save_path: Path to save the plot (if None, only display)
    """
    num_tasks = task_task_accuracies.shape[0]
    avg_accuracies = []
    
    # Calculate average accuracy after training on each task
    for i in range(num_tasks):
        # Get accuracies for all tasks trained so far (tasks 0 to i)
        task_accuracies = task_task_accuracies[i, 0:i+1]
        avg_accuracy = np.mean(task_accuracies)
        avg_accuracies.append(avg_accuracy)
    
    # Plot average accuracies
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, num_tasks + 1), avg_accuracies, 'o-', linewidth=2, markersize=8)
    
    # Add data points with values
    for i, acc in enumerate(avg_accuracies):
        plt.text(i + 1, acc + 0.02, f'{acc:.3f}', ha='center', fontweight='bold')
    
    plt.xlabel('Number of Tasks Trained')
    plt.ylabel('Average Accuracy')
    
    # Set title
    if title:
        plt.title(title)
    else:
        plt.title('Average Accuracy Across All Trained Tasks')
    
    plt.xticks(range(1, num_tasks + 1))
    plt.grid(True)
    plt.ylim(0, 1.05)  # Set y-axis limit to accommodate labels
    
    # Add horizontal line at 1.0 for reference
    plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    
    plt.close()


def plot_all_results(all_accuracies, task_task_accuracies, experiment_name, task_labels=None):
    """
    Plot all results for an experiment.
    
    Args:
        all_accuracies: Dictionary mapping task_id to list of accuracies
        task_task_accuracies: 2D array of accuracies
        experiment_name: Name of the experiment (for filenames and titles)
        task_labels: List of task labels (optional)
    """
    # Create output directory if it doesn't exist
    import os
    os.makedirs('results', exist_ok=True)
    
    # Plot learning curves
    plot_learning_curves(
        all_accuracies,
        title=f'Learning Curves - {experiment_name}',
        save_path=f'results/{experiment_name}_learning_curves.png'
    )
    
    # Plot task-task matrix
    plot_task_matrix(
        task_task_accuracies,
        task_labels=task_labels,
        title=f'Task-Task Accuracy Matrix - {experiment_name}',
        save_path=f'results/{experiment_name}_task_matrix.png'
    )
    
    # Plot average accuracies
    plot_average_accuracies(
        task_task_accuracies,
        title=f'Average Accuracy - {experiment_name}',
        save_path=f'results/{experiment_name}_avg_accuracy.png'
    )
    
    # Save raw data for future analysis
    np.save(f'results/{experiment_name}_task_matrix.npy', task_task_accuracies)
    
    print(f"All plots saved to 'results/' directory")