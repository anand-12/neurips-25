"""
Training utilities for continual learning with Conditional Neural Processes.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import copy
from typing import List, Tuple, Dict, Any, Optional, Union
from tqdm import tqdm

from datasets import select_context_points


#----------------------------------------
# Functional Regularization
#----------------------------------------

def functional_regularization_loss(
    current_model,
    previous_models: List,
    context_sets: List[Tuple[torch.Tensor, torch.Tensor]],
    x_target: torch.Tensor,
    task_id: Optional[int] = None
) -> torch.Tensor:
    """
    Compute functional regularization loss by comparing predictions of the current model
    with predictions from previous task models on their respective domains.
    
    Args:
        current_model: Current model
        previous_models: List of models from previous tasks
        context_sets: List of context sets (x, y) for previous tasks
        x_target: Target input points for regularization
        task_id: Task ID (for multi-head models)
        
    Returns:
        Functional regularization loss
    """
    if not previous_models:
        return torch.tensor(0.0).to(x_target.device)
    
    reg_loss = torch.tensor(0.0).to(x_target.device)
    num_valid_tasks = 0
    
    for i, prev_model in enumerate(previous_models):
        try:
            x_context, y_context = context_sets[i]
            
            # Prepare context in batch format
            x_context_batch = x_context.unsqueeze(0)
            y_context_batch = y_context.unsqueeze(0)
            
            # For CNN models, make sure x_target has the right format
            if len(x_target.shape) == 4 and x_target.shape[1] in [1, 3]:  # [batch, channels, H, W]
                x_target_batch = x_target.unsqueeze(0) if len(x_target.shape) == 4 else x_target
            elif len(x_target.shape) == 5:  # Already in correct format [batch, target, channels, H, W]
                x_target_batch = x_target
            else:
                x_target_batch = x_target.unsqueeze(0) if len(x_target.shape) == 3 else x_target
            
            # Get predictions from previous model
            if hasattr(prev_model, 'num_tasks'):  # Multi-head models
                prev_pred = prev_model.get_functional_prediction(x_context_batch, y_context_batch, x_target_batch, i)
            else:  # Single-head models
                prev_pred = prev_model.get_functional_prediction(x_context_batch, y_context_batch, x_target_batch)
            
            # Get predictions from current model
            if hasattr(current_model, 'num_tasks'):  # Multi-head models
                current_pred = current_model(x_context_batch, y_context_batch, x_target_batch, task_id)
            else:  # Single-head models
                current_pred = current_model(x_context_batch, y_context_batch, x_target_batch)
            
            # Compute KL divergence between distributions
            prev_probs = F.softmax(prev_pred, dim=-1)
            current_probs = F.softmax(current_pred, dim=-1)
            
            # Compute KL divergence
            kl_div = F.kl_div(current_probs.log(), prev_probs, reduction='batchmean')
            reg_loss += kl_div
            num_valid_tasks += 1
        except Exception as e:
            print(f"Warning: Skipping regularization for task {i} due to error: {str(e)}")
            continue
    
    return reg_loss / max(1, num_valid_tasks)


def uncertainty_weighted_functional_regularization(
    current_model,
    previous_models: List,
    context_sets: List[Tuple[torch.Tensor, torch.Tensor]],
    x_target: torch.Tensor,
    task_id: Optional[int] = None
) -> torch.Tensor:
    """
    Uncertainty-weighted functional regularization.
    Weights the regularization by the certainty of previous models' predictions.
    
    Args:
        current_model: Current model
        previous_models: List of models from previous tasks
        context_sets: List of context sets (x, y) for previous tasks
        x_target: Target input points for regularization
        task_id: Task ID (for multi-head models)
        
    Returns:
        Weighted functional regularization loss
    """
    if not previous_models:
        return torch.tensor(0.0).to(x_target.device)
    
    reg_loss = torch.tensor(0.0).to(x_target.device)
    num_valid_tasks = 0
    
    for i, prev_model in enumerate(previous_models):
        try:
            x_context, y_context = context_sets[i]
            
            # Prepare context in batch format
            x_context_batch = x_context.unsqueeze(0)
            y_context_batch = y_context.unsqueeze(0)
            
            # For CNN models, make sure x_target has the right format
            if len(x_target.shape) == 4 and x_target.shape[1] in [1, 3]:  # [batch, channels, H, W]
                x_target_batch = x_target.unsqueeze(0) if len(x_target.shape) == 4 else x_target
            elif len(x_target.shape) == 5:  # Already in correct format [batch, target, channels, H, W]
                x_target_batch = x_target
            else:
                x_target_batch = x_target.unsqueeze(0) if len(x_target.shape) == 3 else x_target
            
            # Get predictions from previous model
            if hasattr(prev_model, 'num_tasks'):  # Multi-head models
                prev_pred = prev_model.get_functional_prediction(x_context_batch, y_context_batch, x_target_batch, i)
            else:  # Single-head models
                prev_pred = prev_model.get_functional_prediction(x_context_batch, y_context_batch, x_target_batch)
            
            # Calculate entropy (uncertainty) of previous model predictions
            prev_probs = F.softmax(prev_pred, dim=-1)
            entropy = -torch.sum(prev_probs * torch.log(prev_probs + 1e-10), dim=-1)
            certainty_weights = torch.exp(-entropy)  # Higher weight for certain predictions
            
            # Get predictions from current model
            if hasattr(current_model, 'num_tasks'):  # Multi-head models
                current_pred = current_model(x_context_batch, y_context_batch, x_target_batch, task_id)
            else:  # Single-head models
                current_pred = current_model(x_context_batch, y_context_batch, x_target_batch)
            
            current_probs = F.softmax(current_pred, dim=-1)
            
            # Weighted KL divergence
            kl_div = F.kl_div(current_probs.log(), prev_probs, reduction='none')
            weighted_kl = (kl_div.sum(-1) * certainty_weights).mean()
            
            reg_loss += weighted_kl
            num_valid_tasks += 1
        except Exception as e:
            print(f"Warning: Skipping weighted regularization for task {i} due to error: {str(e)}")
            continue
    
    return reg_loss / max(1, num_valid_tasks)


#----------------------------------------
# Training Functions
#----------------------------------------

def train_task(
    model: nn.Module,
    task_id: int,
    train_data: torch.Tensor,
    train_labels: torch.Tensor,
    previous_models: List[nn.Module],
    context_sets: List[Tuple[torch.Tensor, torch.Tensor]],
    test_datasets: List[Tuple[torch.Tensor, torch.Tensor]],
    config: Dict[str, Any]
) -> Tuple[nn.Module, Dict[int, List[float]]]:
    """
    Train a model on a specific task with continual learning.
    
    Args:
        model: Model to train
        task_id: ID of the current task
        train_data: Training data for the current task
        train_labels: Training labels for the current task
        previous_models: List of models from previous tasks
        context_sets: List of context sets for previous tasks
        test_datasets: List of test datasets for all tasks
        config: Configuration dictionary with hyperparameters
        
    Returns:
        Tuple of (trained_model, task_accuracies)
    """
    # Extract hyperparameters from config
    batch_size = config.get('batch_size', 128)
    learning_rate = config.get('learning_rate', 0.0005)
    num_epochs = config.get('num_epochs', 5)
    func_reg_weight = config.get('func_reg_weight', 100.0)
    use_weighted_reg = config.get('use_weighted_reg', True)
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Create dataset and data loader
    dataset = TensorDataset(train_data, train_labels)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Track accuracies for all tasks seen so far
    task_accuracies = {i: [] for i in range(task_id + 1)}
    
    # For each epoch
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        total_task_loss = 0.0
        total_reg_loss = 0.0
        num_batches = 0
        
        # Progress bar for training
        pbar = tqdm(train_loader, desc=f"Task {task_id+1}, Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, (x_batch, y_batch) in enumerate(pbar):
            # Skip very small batches
            if len(x_batch) <= 1:
                continue
            
            optimizer.zero_grad()
            
            # Split batch into context and target
            split_idx = max(1, len(x_batch) // 2)
            
            # Handle different input formats
            if len(x_batch.shape) == 2:  # MNIST (flattened)
                # Reshape for CNP format [batch, context_size, x_dim]
                x_context = x_batch[:split_idx].unsqueeze(0)
                y_context = y_batch[:split_idx].unsqueeze(0)
                x_target = x_batch[split_idx:].unsqueeze(0)
                y_target = y_batch[split_idx:].unsqueeze(0)
            elif len(x_batch.shape) == 4:  # CIFAR (images)
                # Reshape for CNP format [batch, context_size, channels, H, W]
                x_context = x_batch[:split_idx].unsqueeze(0)
                y_context = y_batch[:split_idx].unsqueeze(0)
                x_target = x_batch[split_idx:].unsqueeze(0)
                y_target = y_batch[split_idx:].unsqueeze(0)
            else:
                raise ValueError(f"Unexpected batch shape: {x_batch.shape}")
            
            # Forward pass
            if hasattr(model, 'num_tasks'):  # Multi-head models need task_id
                y_pred = model(x_context, y_context, x_target, task_id)
            else:
                y_pred = model(x_context, y_context, x_target)
            
            # Task-specific prediction loss
            y_pred_flat = y_pred.reshape(-1, y_batch.shape[1])
            y_target_flat = torch.argmax(y_target.reshape(-1, y_batch.shape[1]), dim=1)
            task_loss = F.cross_entropy(y_pred_flat, y_target_flat)
            
            # Functional regularization
            reg_loss = torch.tensor(0.0).to(x_batch.device)
            if previous_models:
                if use_weighted_reg:
                    reg_loss = uncertainty_weighted_functional_regularization(
                        model, previous_models, context_sets, x_target,
                        task_id if hasattr(model, 'num_tasks') else None
                    )
                else:
                    reg_loss = functional_regularization_loss(
                        model, previous_models, context_sets, x_target,
                        task_id if hasattr(model, 'num_tasks') else None
                    )
            
            # Total loss
            loss = task_loss + func_reg_weight * reg_loss
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            # Update statistics
            total_loss += loss.item()
            total_task_loss += task_loss.item()
            total_reg_loss += reg_loss.item() if previous_models else 0
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'task_loss': f"{task_loss.item():.4f}",
                'reg_loss': f"{reg_loss.item():.4f}" if previous_models else "N/A"
            })
        
        # Average loss for this epoch
        avg_loss = total_loss / max(1, num_batches)
        avg_task_loss = total_task_loss / max(1, num_batches)
        avg_reg_loss = total_reg_loss / max(1, num_batches) if previous_models else 0
        
        print(f"Task {task_id+1}, Epoch {epoch+1}/{num_epochs}, "
              f"Loss: {avg_loss:.4f}, Task Loss: {avg_task_loss:.4f}, "
              f"Reg Loss: {avg_reg_loss:.4f}")
        
        # Evaluate on current and all previous tasks
        for eval_task_id in range(task_id + 1):
            accuracy = evaluate(
                model, eval_task_id, test_datasets[eval_task_id],
                multi_head=hasattr(model, 'num_tasks')
            )
            task_accuracies[eval_task_id].append(accuracy)
            print(f"  Task {eval_task_id+1} Test Accuracy: {accuracy:.4f}")
    
    return model, task_accuracies


def evaluate(
    model: nn.Module,
    task_id: int,
    test_data: Tuple[torch.Tensor, torch.Tensor],
    multi_head: bool = False,
    batch_size: int = 256
) -> float:
    """
    Evaluate model on a specific task.
    
    Args:
        model: Trained model
        task_id: ID of the task to evaluate
        test_data: Tuple of (x_test, y_test) for the task
        multi_head: Whether the model has multiple heads
        batch_size: Batch size for evaluation
        
    Returns:
        Test accuracy
    """
    model.eval()
    x_test, y_test = test_data
    
    # For large datasets, evaluate in batches
    if len(x_test) > batch_size:
        dataset = TensorDataset(x_test, y_test)
        test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                # Skip small batches
                if len(x_batch) < 5:
                    continue
                
                # Select context points from batch
                context_size = min(100, len(x_batch) // 4)
                x_context, y_context = select_context_points(
                    x_batch, y_batch, context_size, 
                    num_classes=y_batch.shape[1]
                )
                
                # Format context and target
                if len(x_batch.shape) == 2:  # MNIST (flattened)
                    x_context_batch = x_context.unsqueeze(0)
                    y_context_batch = y_context.unsqueeze(0)
                    x_target_batch = x_batch.unsqueeze(0)
                elif len(x_batch.shape) == 4:  # CIFAR (images)
                    x_context_batch = x_context.unsqueeze(0)
                    y_context_batch = y_context.unsqueeze(0)
                    x_target_batch = x_batch.unsqueeze(0)
                else:
                    raise ValueError(f"Unexpected batch shape: {x_batch.shape}")
                
                # Get predictions
                if multi_head:
                    y_pred_batch = model(x_context_batch, y_context_batch, x_target_batch, task_id)
                else:
                    y_pred_batch = model(x_context_batch, y_context_batch, x_target_batch)
                
                y_pred = y_pred_batch.squeeze(0)
                
                # Calculate accuracy
                pred_classes = torch.argmax(y_pred, dim=1)
                true_classes = torch.argmax(y_batch, dim=1)
                
                total_correct += (pred_classes == true_classes).sum().item()
                total_samples += len(true_classes)
        
        return total_correct / max(1, total_samples)
    
    else:
        # For smaller datasets, evaluate all at once
        with torch.no_grad():
            # Select context points (around 10% of the data, balanced)
            context_size = min(100, max(10, int(len(x_test) * 0.1)))
            x_context, y_context = select_context_points(
                x_test, y_test, context_size, 
                num_classes=y_test.shape[1]
            )
            
            # Format context and target
            if len(x_test.shape) == 2:  # MNIST (flattened)
                x_context_batch = x_context.unsqueeze(0)
                y_context_batch = y_context.unsqueeze(0)
                x_target_batch = x_test.unsqueeze(0)
            elif len(x_test.shape) == 4:  # CIFAR (images)
                x_context_batch = x_context.unsqueeze(0)
                y_context_batch = y_context.unsqueeze(0)
                x_target_batch = x_test.unsqueeze(0)
            else:
                raise ValueError(f"Unexpected input shape: {x_test.shape}")
            
            # Get predictions
            if multi_head:
                y_pred_batch = model(x_context_batch, y_context_batch, x_target_batch, task_id)
            else:
                y_pred_batch = model(x_context_batch, y_context_batch, x_target_batch)
            
            y_pred = y_pred_batch.squeeze(0)
            
            # Calculate accuracy
            pred_classes = torch.argmax(y_pred, dim=1)
            true_classes = torch.argmax(y_test, dim=1)
            
            accuracy = (pred_classes == true_classes).float().mean().item()
        
        return accuracy


def run_experiment(
    model_class,
    dataset_fn,
    config: Dict[str, Any],
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
) -> Tuple[nn.Module, Dict[int, List[float]], np.ndarray]:
    """
    Run a complete continual learning experiment.
    
    Args:
        model_class: Model class to use
        dataset_fn: Function to generate datasets
        config: Configuration dictionary with hyperparameters
        device: Device to run on
        
    Returns:
        Tuple of (final_model, all_accuracies, task_task_accuracies)
    """
    # Extract hyperparameters
    num_tasks = config.get('num_tasks', 5)
    context_points_per_task = config.get('context_points_per_task', 200)
    model_params = config.get('model_params', {})
    data_dir = config.get('data_dir', './data')
    
    # Generate datasets
    print(f"Generating datasets for {num_tasks} tasks...")
    all_datasets = dataset_fn(num_tasks, device, data_dir)
    
    # Initialize model
    model = model_class(**model_params).to(device)
    
    # Track previous models and context sets for functional regularization
    previous_models = []
    context_sets = []
    
    # Track accuracies for all tasks
    all_accuracies = {i: [] for i in range(num_tasks)}
    
    # Create a matrix to track the accuracy of each task after training on each task
    task_task_accuracies = np.zeros((num_tasks, num_tasks))
    
    for task_id in range(num_tasks):
        print(f"\n=== Training on Task {task_id+1}/{num_tasks} ===")
        
        # Get data for current task
        train_data, train_labels, test_data, test_labels = all_datasets[task_id]
        
        # Train on current task
        model, task_accuracies = train_task(
            model, 
            task_id, 
            train_data, 
            train_labels, 
            previous_models, 
            context_sets,
            [(test_data, test_labels) for train_data, train_labels, test_data, test_labels in all_datasets],
            config
        )
        
        # Update accuracies
        for eval_task_id, accuracies in task_accuracies.items():
            all_accuracies[eval_task_id].extend(accuracies)
            # Store the final accuracy for this task after training
            task_task_accuracies[task_id, eval_task_id] = accuracies[-1]
        
        # Evaluate model on all previous tasks to track forgetting
        for eval_task_id in range(task_id):
            eval_data, eval_labels = all_datasets[eval_task_id][2:4]  # Get test data for this task
            accuracy = evaluate(
                model, eval_task_id, (eval_data, eval_labels),
                multi_head=hasattr(model, 'num_tasks')
            )
            # Store the accuracy on previous tasks
            task_task_accuracies[task_id, eval_task_id] = accuracy
            print(f"  After Task {task_id+1}: Accuracy on Task {eval_task_id+1}: {accuracy:.4f}")
        
        # Store model for this task
        task_model = copy.deepcopy(model)
        previous_models.append(task_model)
        
        # Select and store context points for this task
        x_context, y_context = select_context_points(
            train_data, train_labels, context_points_per_task,
            num_classes=train_labels.shape[1]
        )
        context_sets.append((x_context, y_context))
    
    return model, all_accuracies, task_task_accuracies


#----------------------------------------
# Visualization Functions
#----------------------------------------

def plot_learning_curves(all_accuracies: Dict[int, List[float]], title: str = None, save_path: str = None):
    """Plot learning curves for all tasks across epochs."""
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 6))
    
    for task_id, accuracies in all_accuracies.items():
        epochs = range(1, len(accuracies) + 1)
        plt.plot(epochs, accuracies, label=f'Task {task_id+1}')
    
    plt.xlabel('Epochs')
    plt.ylabel('Test Accuracy')
    plt.title(title or 'Test Accuracy for All Tasks During Training')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    
    plt.close()


def plot_task_matrix(task_task_accuracies: np.ndarray, task_labels: List[str] = None, 
                    title: str = None, save_path: str = None):
    """Plot the task-task accuracy matrix."""
    import matplotlib.pyplot as plt
    
    num_tasks = task_task_accuracies.shape[0]
    
    # Default task labels if not provided
    if task_labels is None:
        task_labels = [f"Task {i+1}" for i in range(num_tasks)]
    
    plt.figure(figsize=(10, 8))
    plt.imshow(task_task_accuracies, cmap='magma', vmin=0, vmax=1)
    plt.colorbar(label='Accuracy')
    plt.xlabel('Evaluated on Task')
    plt.ylabel('After Training on Task')
    plt.title(title or 'Task-Task Accuracy Matrix')
    
    plt.xticks(range(num_tasks), task_labels, rotation=45, ha="right")
    plt.yticks(range(num_tasks), task_labels)
    
    # Add accuracy values to cells
    for i in range(num_tasks):
        for j in range(num_tasks):
            if i >= j:  # Only show values for trained tasks
                plt.text(j, i, f'{task_task_accuracies[i, j]:.2f}', 
                         ha='center', va='center', 
                         color='white' if task_task_accuracies[i, j] < 0.7 else 'black')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.close()


def plot_average_accuracies(task_task_accuracies: np.ndarray, title: str = None, save_path: str = None):
    """Plot the average accuracy after training on each task."""
    import matplotlib.pyplot as plt
    
    num_tasks = task_task_accuracies.shape[0]
    avg_accuracies = []
    
    # Calculate average accuracy after training on each task
    for i in range(num_tasks):
        # Get accuracies for all tasks trained so far (tasks 0 to i)
        task_accuracies = task_task_accuracies[i, 0:i+1]
        avg_accuracy = np.mean(task_accuracies)
        avg_accuracies.append(avg_accuracy)
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_tasks + 1), avg_accuracies, 'o-', linewidth=2, markersize=8)
    
    # Add data points with values
    for i, acc in enumerate(avg_accuracies):
        plt.text(i + 1, acc + 0.02, f'{acc:.3f}', ha='center', fontweight='bold')
    
    plt.xlabel('Number of Tasks Trained')
    plt.ylabel('Average Accuracy')
    plt.title(title or 'Average Accuracy Across All Trained Tasks')
    plt.xticks(range(1, num_tasks + 1))
    plt.grid(True)
    plt.ylim(0, 1.05)
    
    # Add horizontal line at 1.0 for reference
    plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.close()


def plot_results(all_accuracies, task_task_accuracies, experiment_name, task_labels=None):
    """Plot all results for an experiment."""
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