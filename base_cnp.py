import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import time

# Set random seeds for reproducibility (optional, but good for comparison)
# torch.manual_seed(42)
# np.random.seed(42)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Data Loading and Permutation ---
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
mnist_train_full = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
mnist_test_full = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Convert to numpy arrays for easier permutation handling
x_train_np = mnist_train_full.data.view(-1, 28*28).float() / 255.0
y_train_np = mnist_train_full.targets
x_test_np = mnist_test_full.data.view(-1, 28*28).float() / 255.0
y_test_np = mnist_test_full.targets

num_tasks = 10 # Reduced for quicker demonstration
permutations = []
task_data_train = []
task_data_test = []

for i in range(num_tasks):
    perm = np.random.permutation(28*28)
    permutations.append(perm)
    
    x_train_perm = x_train_np[:, perm]
    x_test_perm = x_test_np[:, perm]
    
    task_data_train.append((x_train_perm, y_train_np))
    task_data_test.append((x_test_perm, y_test_np))

# --- CNP Model Components ---
class NPEncoder(nn.Module): # This can be reused as is for CNP
    """Encodes (x, y) pairs into a representation r_i."""
    def __init__(self, input_dim_x, input_dim_y, hidden_dim, r_dim):
        super(NPEncoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim_x + input_dim_y, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, r_dim)
        )

    def forward(self, x, y):
        # x: (batch_size, num_points, x_dim)
        # y: (batch_size, num_points, y_dim) - y should be one-hot encoded
        xy_cat = torch.cat([x, y], dim=-1)
        return self.fc(xy_cat)

class CNPDecoder(nn.Module):
    """Decodes (r_aggregated, x_target) into y_target predictions."""
    def __init__(self, x_dim, r_dim, hidden_dim, y_dim_out): # Takes r_dim instead of z_dim
        super(CNPDecoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(r_dim + x_dim, hidden_dim), # Input is r_aggregated + x_target
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, y_dim_out) # Outputting logits for classification
        )

    def forward(self, r_aggregated, x_target):
        # r_aggregated: (batch_size, r_dim)
        # x_target: (batch_size, num_target_points, x_dim)
        
        num_target_points = x_target.size(1)
        # Repeat r_aggregated for each target point
        r_repeated = r_aggregated.unsqueeze(1).repeat(1, num_target_points, 1) # (batch, num_target, r_dim)
        
        rx_cat = torch.cat([r_repeated, x_target], dim=-1)
        return self.fc(rx_cat)


class ConditionalNeuralProcess(nn.Module):
    def __init__(self, x_dim=784, y_dim_onehot=10, r_dim=128, enc_hidden_dim=128, dec_hidden_dim=128, y_dim_out=10):
        super(ConditionalNeuralProcess, self).__init__()
        self.x_dim = x_dim
        self.y_dim_onehot = y_dim_onehot
        self.r_dim = r_dim # Storing r_dim for potential use (e.g. zero-context handling)
        self.y_dim_out = y_dim_out

        self.xy_encoder = NPEncoder(x_dim, y_dim_onehot, enc_hidden_dim, r_dim)
        # No LatentEncoder for CNP
        self.decoder = CNPDecoder(x_dim, r_dim, dec_hidden_dim, y_dim_out) # Decoder takes r_dim

    def aggregate(self, r_i):
        # r_i: (batch_size, num_context_points, r_dim)
        # If r_i is empty (num_context_points=0), mean will be NaN. This must be handled in forward.
        return torch.mean(r_i, dim=1) # (batch_size, r_dim)

    # No reparameterize method for CNP

    def forward(self, x_context, y_context_onehot, x_target):
        # x_context: (batch_size, num_context, x_dim)
        # y_context_onehot: (batch_size, num_context, y_dim_onehot)
        # x_target: (batch_size, num_target, x_dim)

        if x_context.size(1) > 0: # If there are context points
            r_i_context = self.xy_encoder(x_context, y_context_onehot) # (batch, num_context, r_dim)
            r_aggregated = self.aggregate(r_i_context) # (batch, r_dim)
        else: # No context points, use a default representation (e.g., zeros)
            batch_size = x_target.size(0) # Assuming x_target is always present for prediction
            r_aggregated = torch.zeros(batch_size, self.r_dim, device=x_target.device)
        
        y_pred_logits = self.decoder(r_aggregated, x_target) # (batch, num_target, y_dim_out)
        
        return y_pred_logits # CNP doesn't return z_mean, z_logvar

# --- Loss Function (CNP) ---
def cnp_loss(y_pred_logits, y_target_labels):
    # y_pred_logits: (batch, num_target, y_dim_out)
    # y_target_labels: (batch, num_target) - class indices
    
    # Reconstruction Loss (Cross-Entropy)
    # Reshape for cross_entropy: (batch * num_target, y_dim_out) and (batch * num_target)
    ce_loss = F.cross_entropy(y_pred_logits.view(-1, y_pred_logits.size(-1)), 
                              y_target_labels.view(-1), reduction='mean')
    
    # No KL Divergence for CNP
    return ce_loss

# --- Training and Evaluation Helper ---
def get_context_target_split(x_batch, y_batch, num_classes, num_context_range=(5, 50), num_extra_target_range=(10,100)):
    batch_size, total_points, _ = x_batch.shape # x_batch is (1, N, dim)
    
    num_context = np.random.randint(num_context_range[0], min(num_context_range[1] + 1, total_points))
    
    max_target_points = total_points - num_context
    if max_target_points <= 0:
        num_context = total_points - 1
        if num_context < 0: num_context = 0 # handle total_points = 0 or 1
        max_target_points = total_points - num_context


    num_target = np.random.randint(num_extra_target_range[0], min(num_extra_target_range[1] + 1, max_target_points + 1))
    if num_target == 0 and max_target_points > 0: num_target = 1
    if total_points == 0: # Should ideally not happen if DataLoader has items
        x_context = torch.empty(batch_size, 0, x_batch.size(2), device=x_batch.device, dtype=x_batch.dtype)
        y_context_labels = torch.empty(batch_size, 0, device=y_batch.device, dtype=y_batch.dtype)
        x_target = torch.empty(batch_size, 0, x_batch.size(2), device=x_batch.device, dtype=x_batch.dtype)
        y_target_labels = torch.empty(batch_size, 0, device=y_batch.device, dtype=y_batch.dtype)
        y_context_onehot = torch.empty(batch_size, 0, num_classes, device=x_batch.device, dtype=torch.float)
        return x_context, y_context_onehot, y_context_labels, x_target, y_target_labels, False

    indices = torch.randperm(total_points, device=x_batch.device) # Ensure indices are on the same device
    context_indices = indices[:num_context]
    # Ensure target_indices are within bounds
    target_end_idx = min(num_context + num_target, total_points)
    target_indices = indices[num_context : target_end_idx]


    x_context = x_batch[:, context_indices, :]
    y_context_labels = y_batch[:, context_indices]
    
    x_target = x_batch[:, target_indices, :]
    y_target_labels = y_batch[:, target_indices]

    if num_context > 0:
        y_context_onehot = F.one_hot(y_context_labels, num_classes=num_classes).float()
    else:
        y_context_onehot = torch.empty(batch_size, 0, num_classes, device=x_batch.device, dtype=torch.float)
    
    return x_context, y_context_onehot, y_context_labels, x_target, y_target_labels, num_context > 0


def train_cnp_task(model, optimizer, task_id, epochs=5, batch_size=64, num_classes_mnist=10): # Removed kl_weight
    x_train_task_np, y_train_task_np = task_data_train[task_id]
    
    train_dataset = TensorDataset(x_train_task_np, y_train_task_np)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True) # drop_last to ensure consistent batch sizes for splitting
    
    model.train()
    print(f"Training on Task {task_id+1}...")
    for epoch in range(epochs):
        total_loss_epoch = 0
        correct_preds = 0
        total_target_points = 0
        
        for batch_idx, (x_batch_flat, y_batch_labels_flat) in enumerate(train_loader):
            x_batch_flat, y_batch_labels_flat = x_batch_flat.to(device), y_batch_labels_flat.to(device)
            
            current_batch_size = x_batch_flat.size(0)
            # Reshape to (1, current_batch_size, dim) for get_context_target_split
            x_for_split = x_batch_flat.view(1, current_batch_size, -1) 
            y_for_split = y_batch_labels_flat.view(1, current_batch_size)   

            x_context, y_context_onehot, _, x_target, y_target_labels, has_context = \
                get_context_target_split(x_for_split, y_for_split, num_classes=num_classes_mnist,
                                         num_context_range=(max(1, int(current_batch_size * 0.1)), int(current_batch_size * 0.7)),
                                         num_extra_target_range=(max(1, int(current_batch_size * 0.2)), int(current_batch_size * 0.9)))

            if x_target.size(1) == 0: # Skip if no target points
                # print(f"Skipping batch due to no target points. Context points: {x_context.size(1)}")
                continue
            # For CNP, we can proceed even if has_context is False, as the model handles it.
            # However, if we want to ensure context is always used for training representation learning:
            if not has_context and model.training: # Optionally enforce context during training
                 # print(f"Skipping batch due to no context points during training. Target points: {x_target.size(1)}")
                 continue


            optimizer.zero_grad()
            y_pred_logits = model(x_context, y_context_onehot, x_target)
            
            loss = cnp_loss(y_pred_logits, y_target_labels) # Use cnp_loss
            
            loss.backward()
            optimizer.step()
            
            total_loss_epoch += loss.item()
            
            _, predicted_labels = torch.max(y_pred_logits.data, -1)
            correct_preds += (predicted_labels == y_target_labels).sum().item()
            total_target_points += y_target_labels.numel()

        avg_loss = total_loss_epoch / (batch_idx + 1) if (batch_idx +1) > 0 else 0
        accuracy = 100. * correct_preds / total_target_points if total_target_points > 0 else 0
        print(f'  Epoch {epoch+1}/{epochs}: Avg Loss: {avg_loss:.4f}, Acc: {accuracy:.2f}%')


def evaluate_cnp_task(model, task_id, num_eval_batches=50, batch_size=64, fixed_num_context=20, num_classes_mnist=10):
    x_test_task_np, y_test_task_np = task_data_test[task_id]
    test_dataset = TensorDataset(x_test_task_np, y_test_task_np)
    # drop_last=True if fixed_num_context relies on full batch_size, otherwise not strictly needed for eval
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True) 

    model.eval()
    total_correct = 0
    total_targets = 0
    
    print(f"Evaluating on Task {task_id+1}...")
    with torch.no_grad():
        for i, (x_batch_flat, y_batch_labels_flat) in enumerate(test_loader):
            if i >= num_eval_batches:
                break
            x_batch_flat, y_batch_labels_flat = x_batch_flat.to(device), y_batch_labels_flat.to(device)
            
            current_batch_size = x_batch_flat.size(0)
            if current_batch_size == 0: continue

            x_for_split = x_batch_flat.view(1, current_batch_size, -1) 
            y_for_split = y_batch_labels_flat.view(1, current_batch_size)

            num_context_eval = min(fixed_num_context, current_batch_size -1) # ensure at least 1 target point
            if num_context_eval < 0: num_context_eval = 0 # if current_batch_size is 1 or 0

            num_target_eval = current_batch_size - num_context_eval
            if num_target_eval <= 0 : continue 

            indices = torch.randperm(current_batch_size, device=device)
            context_indices = indices[:num_context_eval]
            target_indices = indices[num_context_eval : num_context_eval + num_target_eval]

            x_context_eval = x_for_split[:, context_indices, :]
            y_context_labels_eval = y_for_split[:, context_indices]
            x_target_eval = x_for_split[:, target_indices, :]
            y_target_labels_eval = y_for_split[:, target_indices]
            
            if y_target_labels_eval.numel() == 0: continue


            if num_context_eval > 0:
                y_context_onehot_eval = F.one_hot(y_context_labels_eval, num_classes=num_classes_mnist).float()
            else: # num_context_eval == 0
                y_context_onehot_eval = torch.empty(1, 0, num_classes_mnist, device=device, dtype=torch.float)
            
            y_pred_logits = model(x_context_eval, y_context_onehot_eval, x_target_eval)

            _, predicted = torch.max(y_pred_logits.data, -1)
            total_correct += (predicted == y_target_labels_eval).sum().item()
            total_targets += y_target_labels_eval.numel()
            
    accuracy = 100. * total_correct / total_targets if total_targets > 0 else 0
    return accuracy


# --- Run Continual Learning Experiment (CNP) ---
def run_continual_learning_cnp_experiment():
    # CNP Hyperparameters
    x_dim = 784
    y_dim_onehot = 10 
    r_dim = 256       # Representation dim for each (x,y) pair, also for aggregated r
    # z_dim is not used in CNP
    enc_hidden_dim = 256
    dec_hidden_dim = 256
    y_dim_out = 10    # Output logits for 10 classes
    num_classes_mnist = 10 # Explicitly pass to functions needing it

    model = ConditionalNeuralProcess(x_dim, y_dim_onehot, r_dim, enc_hidden_dim, dec_hidden_dim, y_dim_out).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4) # CNPs can also be sensitive to LR
    
    epochs_per_task = 10 
    batch_size_train = 32  # Smaller batch for more context/target splits

    task_accuracies = {i: [] for i in range(num_tasks)}
    avg_accuracies_over_time = []

    for current_task_idx in range(num_tasks):
        print(f"\n--- Training on Task {current_task_idx+1}/{num_tasks} (CNP) ---")
        train_cnp_task(model, optimizer, current_task_idx, epochs=epochs_per_task, batch_size=batch_size_train, num_classes_mnist=num_classes_mnist)
        
        current_eval_accuracies = []
        print(f"\n--- Evaluating after Task {current_task_idx+1} (CNP) ---")
        for eval_task_idx in range(num_tasks):
            if eval_task_idx <= current_task_idx: # Evaluate on current and all previous tasks
                acc = evaluate_cnp_task(model, eval_task_idx, num_eval_batches=30, batch_size=64, fixed_num_context=30, num_classes_mnist=num_classes_mnist)
                task_accuracies[eval_task_idx].append(acc)
                current_eval_accuracies.append(acc)
                print(f"  Accuracy on Task {eval_task_idx+1}: {acc:.2f}%")
            else:
                task_accuracies[eval_task_idx].append(np.nan) 
        
        if current_eval_accuracies:
             avg_acc = np.mean([acc for acc in current_eval_accuracies if not np.isnan(acc)])
             avg_accuracies_over_time.append(avg_acc)
             print(f"  Average accuracy over seen tasks: {avg_acc:.2f}%")

    return task_accuracies, avg_accuracies_over_time

# --- Run and Plot ---
print("Starting Conditional Neural Process Continual Learning experiment...")
task_accuracies_cnp, avg_accuracies_cnp = run_continual_learning_cnp_experiment()

# Plotting the results
plt.figure(figsize=(12, 7))
for task_id in range(num_tasks):
    # accuracies for task_id are recorded after task_id, task_id+1, ..., num_tasks-1 are trained
    # so the list task_accuracies_cnp[task_id] has num_tasks - task_id elements
    
    # x-axis stages: 1 to num_tasks
    stages = list(range(1, num_tasks + 1))
    
    # Pad with NaNs at the beginning for tasks not yet introduced
    # The first accuracy for task_id is recorded *after* training on task_id (stage task_id+1)
    plot_accuracies = [np.nan] * task_id 
    plot_accuracies.extend(task_accuracies_cnp[task_id])
    # Ensure it's the correct length for all stages by padding end if necessary (should match num_tasks-task_id elements)
    # The list task_accuracies_cnp[task_id] should have (num_tasks - task_id) entries
    # So after padding, len(plot_accuracies) = task_id + (num_tasks - task_id) = num_tasks
    
    plt.plot(stages, plot_accuracies, marker='o', linestyle='-', label=f'Task {task_id+1} Perf.')

plt.plot(range(1, num_tasks + 1), avg_accuracies_cnp, marker='x', linestyle='--', color='black', label='Avg. Accuracy (Seen Tasks)')
plt.xlabel('Training Stage (After Training on Task X)')
plt.ylabel('Accuracy (%)')
plt.title('Conditional Neural Process: Continual Learning on Permuted MNIST')
plt.xticks(range(1, num_tasks + 1))
plt.legend(loc='best')
plt.grid(True)
plt.ylim(0, 101)
plt.savefig('cnp_catastrophic_forgetting.png')
plt.show()


print("\n--- Final CNP Accuracies Table (Performance on Task Y after training on Task X) ---")
header = "Trained on Task -> |"
for i in range(num_tasks):
    header += f"   Task {i+1}   |"
print(header)
print("-" * len(header))

for eval_task_id in range(num_tasks):
    row_str = f"Perf. on Task {eval_task_id+1:2d}  | "
    for train_stage_idx in range(num_tasks): # Corresponds to training stage 0 to num_tasks-1
        if train_stage_idx < eval_task_id:
            # This task (eval_task_id) has not been trained yet by this stage (train_stage_idx)
            row_str += "    -     | "
        else:
            # task_accuracies_cnp[eval_task_id] stores accuracies *after* each relevant training stage.
            # The first accuracy for eval_task_id is after training eval_task_id.
            # Its index in the list is (train_stage_idx - eval_task_id).
            acc_list_for_eval_task = task_accuracies_cnp[eval_task_id]
            current_acc_idx = train_stage_idx - eval_task_id
            if current_acc_idx < len(acc_list_for_eval_task):
                acc_val = acc_list_for_eval_task[current_acc_idx]
                row_str += f"{acc_val:6.2f}   | " if not np.isnan(acc_val) else "  N/A   | "
            else:
                row_str += "  ERR   | " # Should not happen if logic is correct
    print(row_str)


print("\nFinal average accuracy for CNP across tasks seen so far, after all training:")
if avg_accuracies_cnp:
    print(f"{avg_accuracies_cnp[-1]:.2f}%")