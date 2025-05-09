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

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Data Loading and Permutation (same as before) ---
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
mnist_train_full = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
mnist_test_full = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

x_train_np = mnist_train_full.data.view(-1, 28*28).float() / 255.0
y_train_np = mnist_train_full.targets
x_test_np = mnist_test_full.data.view(-1, 28*28).float() / 255.0
y_test_np = mnist_test_full.targets

# GLOBAL num_tasks for model and experiment
NUM_TASKS = 3 # Reduced for quicker demonstration; can be increased

permutations = []
task_data_train = []
task_data_test = []

for i in range(NUM_TASKS):
    perm = np.random.permutation(28*28)
    permutations.append(perm)
    
    x_train_perm = x_train_np[:, perm]
    x_test_perm = x_test_np[:, perm]
    
    task_data_train.append((x_train_perm, y_train_np))
    task_data_test.append((x_test_perm, y_test_np))

# --- Neural Process Model with Task-Specific Priors ---
class NPEncoder(nn.Module): # Same as before
    def __init__(self, input_dim_x, input_dim_y, hidden_dim, r_dim):
        super(NPEncoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim_x + input_dim_y, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, r_dim)
        )
    def forward(self, x, y):
        xy_cat = torch.cat([x, y], dim=-1)
        return self.fc(xy_cat)

class LatentEncoder(nn.Module): # Same as before
    def __init__(self, r_dim, z_dim):
        super(LatentEncoder, self).__init__()
        self.fc_mean = nn.Linear(r_dim, z_dim)
        self.fc_logvar = nn.Linear(r_dim, z_dim)
    def forward(self, r_aggregated):
        z_mean = self.fc_mean(r_aggregated)
        z_logvar = self.fc_logvar(r_aggregated)
        return z_mean, z_logvar

class NPDecoder(nn.Module): # Same as before
    def __init__(self, x_dim, z_dim, hidden_dim, y_dim_out):
        super(NPDecoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(z_dim + x_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, y_dim_out)
        )
    def forward(self, z_sample, x_target):
        num_target_points = x_target.size(1)
        z_repeated = z_sample.unsqueeze(1).repeat(1, num_target_points, 1)
        zx_cat = torch.cat([z_repeated, x_target], dim=-1)
        return self.fc(zx_cat)

class NeuralProcessDynamicPrior(nn.Module):
    def __init__(self, num_all_tasks, x_dim=784, y_dim_onehot=10, r_dim=128, z_dim=64, 
                 enc_hidden_dim=128, dec_hidden_dim=128, y_dim_out=10):
        super(NeuralProcessDynamicPrior, self).__init__()
        self.x_dim = x_dim
        self.y_dim_onehot = y_dim_onehot
        self.y_dim_out = y_dim_out
        self.z_dim = z_dim
        self.num_all_tasks = num_all_tasks

        self.xy_encoder = NPEncoder(x_dim, y_dim_onehot, enc_hidden_dim, r_dim)
        self.latent_encoder = LatentEncoder(r_dim, z_dim) # This gives q(z|D_context)
        self.decoder = NPDecoder(x_dim, z_dim, dec_hidden_dim, y_dim_out)

        # Learnable prior parameters for each task
        # Initialize prior means randomly, logvars to 0 (variance 1)
        self.task_prior_means = nn.Parameter(torch.randn(num_all_tasks, z_dim) * 0.1)
        self.task_prior_logvars = nn.Parameter(torch.zeros(num_all_tasks, z_dim))

    def aggregate(self, r_i):
        return torch.mean(r_i, dim=1)

    def reparameterize(self, z_mean, z_logvar):
        std = torch.exp(0.5 * z_logvar)
        eps = torch.randn_like(std)
        return z_mean + eps * std

    def get_task_prior(self, task_id):
        if not (0 <= task_id < self.num_all_tasks):
            raise ValueError(f"Task ID {task_id} is out of bounds for {self.num_all_tasks} tasks.")
        return self.task_prior_means[task_id], self.task_prior_logvars[task_id]

    def forward(self, x_context, y_context_onehot, x_target):
        # This part defines q(z|context_data)
        r_i_context = self.xy_encoder(x_context, y_context_onehot)
        r_aggregated = self.aggregate(r_i_context)
        
        z_q_mean, z_q_logvar = self.latent_encoder(r_aggregated) # Parameters for q(z|D_c)
        z_sample = self.reparameterize(z_q_mean, z_q_logvar)
        
        y_pred_logits = self.decoder(z_sample, x_target)
        
        # Return parameters of q(z) for KL divergence calculation against the task-specific prior
        return y_pred_logits, z_q_mean, z_q_logvar

# --- Modified Loss Function ---
def np_loss_dynamic_prior(y_pred_logits, y_target_labels, 
                          z_q_mean, z_q_logvar, 
                          task_prior_mean, task_prior_logvar, 
                          kl_weight=0.1):
    # Reconstruction Loss (Cross-Entropy)
    ce_loss = F.cross_entropy(y_pred_logits.view(-1, y_pred_logits.size(-1)), 
                              y_target_labels.view(-1), reduction='mean')
    
    # KL Divergence D_KL(q(z|D_c) || p_k(z))
    # q ~ N(z_q_mean, exp(z_q_logvar))
    # p_k ~ N(task_prior_mean, exp(task_prior_logvar))
    # KL = 0.5 * sum_i ( (var_q_i + (mean_q_i - mean_p_i)^2) / var_p_i - 1 + log_var_p_i - log_var_q_i )
    
    var_q = torch.exp(z_q_logvar)
    var_p = torch.exp(task_prior_logvar) # Make sure this is not zero
    
    # Add a small epsilon to var_p to prevent division by zero if logvar is too small
    var_p = var_p + 1e-7 

    kl_div_terms = var_q / var_p + \
                   (z_q_mean - task_prior_mean).pow(2) / var_p - \
                   1 + \
                   task_prior_logvar - \
                   z_q_logvar
    kl_div = 0.5 * torch.sum(kl_div_terms, dim=1) # Sum over z_dim
    kl_loss = torch.mean(kl_div) # Average over batch
    
    total_loss = ce_loss + kl_weight * kl_loss
    return total_loss, ce_loss, kl_loss

# --- Training and Evaluation Helper (get_context_target_split is the same) ---
def get_context_target_split(x_batch, y_batch, num_context_range=(5, 50), num_extra_target_range=(10,100), min_context=1, min_target=1):
    batch_size, total_points_in_batch, _ = x_batch.shape # x_batch is (1, num_images_in_original_batch, x_dim)
    
    if total_points_in_batch < min_context + min_target:
        # Not enough points to satisfy min_context and min_target, handle gracefully
        # This might happen if batch_size is very small.
        # Option 1: Skip this batch (problematic if many such batches)
        # Option 2: Use all possible as context, and remaining (if any) as target, or vice-versa.
        # For now, let's try to enforce. If not possible, could lead to errors if num_context or num_target is 0.
        if total_points_in_batch <= min_context : # Use all as context, no target (problem for loss)
             # Let's ensure we have at least one target
            if total_points_in_batch > 0:
                num_target = 1
                num_context = total_points_in_batch - 1
                if num_context < 0: num_context = 0 # only 1 point, make it target
            else: # No points at all
                return None, None, None, None, None, False
        else: # Enough for min_context, use rest for target
            num_context = min_context
            num_target = total_points_in_batch - num_context


    else:
        low_ctx = max(min_context, num_context_range[0])
        high_ctx = min(num_context_range[1] + 1, total_points_in_batch - min_target +1)
        if low_ctx >= high_ctx: # Range is invalid, e.g. num_context_range[0] too high
            num_context = min(max(min_context, total_points_in_batch - min_target), num_context_range[1]) # take a sensible val
        else:
            num_context = np.random.randint(low_ctx, high_ctx)

        remaining_points = total_points_in_batch - num_context
        low_tgt = max(min_target, num_extra_target_range[0])
        high_tgt = min(num_extra_target_range[1] + 1, remaining_points + 1)

        if low_tgt >= high_tgt:
            num_target = min(max(min_target, remaining_points), num_extra_target_range[1])
        else:
            num_target = np.random.randint(low_tgt, high_tgt)
    
    if num_context == 0 and num_target == 0 : # Should not happen if total_points_in_batch > 0
        return None, None, None, None, None, False


    indices = torch.randperm(total_points_in_batch, device=x_batch.device) # Ensure indices are on same device
    context_indices = indices[:num_context]
    target_indices = indices[num_context : num_context + num_target]

    x_context = x_batch[:, context_indices, :]
    y_context_labels = y_batch[:, context_indices]
    
    x_target = x_batch[:, target_indices, :]
    y_target_labels = y_batch[:, target_indices]

    y_context_onehot = F.one_hot(y_context_labels, num_classes=10).float()
    
    return x_context, y_context_onehot, y_context_labels, x_target, y_target_labels, num_context > 0 and num_target > 0


def train_np_task(model, optimizer, current_task_id, epochs=5, batch_size=64, kl_weight=0.01):
    x_train_task_np, y_train_task_np = task_data_train[current_task_id]
    train_dataset = TensorDataset(x_train_task_np, y_train_task_np)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True) 
    
    model.train()
    print(f"Training on Task {current_task_id+1} (Prior Index {current_task_id})...")

    for epoch in range(epochs):
        total_loss_epoch, total_ce_epoch, total_kl_epoch = 0, 0, 0
        correct_preds, total_target_points = 0, 0
        num_processed_batches = 0 # Counter for batches that actually run
        
        for batch_idx, (x_batch_flat, y_batch_labels_flat) in enumerate(train_loader):
            x_batch_flat, y_batch_labels_flat = x_batch_flat.to(device), y_batch_labels_flat.to(device)
            
            current_bs_for_split = x_batch_flat.size(0)
            x_for_split = x_batch_flat.view(1, current_bs_for_split, -1)
            y_for_split = y_batch_labels_flat.view(1, current_bs_for_split)

            min_total_points_needed = 5 
            if current_bs_for_split < min_total_points_needed:
                continue

            split_results = get_context_target_split(
                x_for_split, y_for_split,
                num_context_range=(max(1, int(current_bs_for_split * 0.2)), int(current_bs_for_split * 0.7)),
                num_extra_target_range=(max(1, int(current_bs_for_split * 0.3)), int(current_bs_for_split * 0.8)),
                min_context=2, min_target=3 
            )
            if split_results is None or not split_results[-1]: continue 
            x_context, y_context_onehot, _, x_target, y_target_labels, valid_split = split_results
            if not valid_split: continue

            optimizer.zero_grad()
            y_pred_logits, z_q_mean, z_q_logvar = model(x_context, y_context_onehot, x_target)
            
            # *** CORRECTED CALL SITE ***
            loss, ce, kl = np_loss_dynamic_prior(
                y_pred_logits, 
                y_target_labels, 
                z_q_mean, 
                z_q_logvar,
                task_prior_mean=model.task_prior_means[current_task_id],    # Explicit keyword
                task_prior_logvar=model.task_prior_logvars[current_task_id], # Explicit keyword
                kl_weight=kl_weight  # kl_weight from train_np_task's arguments
            )
            
            loss.backward()
            optimizer.step()
            
            total_loss_epoch += loss.item()
            total_ce_epoch += ce.item()
            total_kl_epoch += kl.item()
            
            _, predicted_labels = torch.max(y_pred_logits.data, -1)
            correct_preds += (predicted_labels == y_target_labels).sum().item()
            total_target_points += y_target_labels.numel()
            num_processed_batches += 1

        if num_processed_batches > 0 : 
            avg_loss = total_loss_epoch / num_processed_batches
            avg_ce = total_ce_epoch / num_processed_batches
            avg_kl = total_kl_epoch / num_processed_batches
            accuracy = 100. * correct_preds / total_target_points if total_target_points > 0 else 0
            print(f'  Epoch {epoch+1}/{epochs}: Avg Loss: {avg_loss:.4f} (CE: {avg_ce:.4f}, KL: {avg_kl:.4f}), Acc: {accuracy:.2f}%')
        elif epoch == epochs - 1 : # If all batches were skipped, print for the last epoch
            print(f'  Epoch {epoch+1}/{epochs}: No batches processed this epoch.')

def evaluate_np_task(model, eval_task_id, num_eval_batches=50, batch_size=64, fixed_num_context=20):
    x_test_task_np, y_test_task_np = task_data_test[eval_task_id]
    test_dataset = TensorDataset(x_test_task_np, y_test_task_np)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    model.eval()
    total_correct, total_targets = 0, 0
    
    print(f"Evaluating on Task {eval_task_id+1} (Prior Index {eval_task_id})...")
    # During evaluation, the prior is fixed (it's what was learned).
    # The KL divergence isn't part of accuracy calculation, but q(z) characteristics are shaped by this prior.
    
    with torch.no_grad():
        for i, (x_batch_flat, y_batch_labels_flat) in enumerate(test_loader):
            if i >= num_eval_batches: break
            x_batch_flat, y_batch_labels_flat = x_batch_flat.to(device), y_batch_labels_flat.to(device)
            
            current_bs_for_split = x_batch_flat.size(0)
            if current_bs_for_split < 5: continue # Min points for split

            x_for_split = x_batch_flat.view(1, current_bs_for_split, -1) 
            y_for_split = y_batch_labels_flat.view(1, current_bs_for_split)

            num_context_eval = min(fixed_num_context, current_bs_for_split -1)
            if num_context_eval < 1 and current_bs_for_split > 1: num_context_eval = 1
            if current_bs_for_split <= 1: num_context_eval = 0 
            num_target_eval = current_bs_for_split - num_context_eval
            if num_target_eval <=0 : continue

            indices = torch.randperm(current_bs_for_split, device=x_batch_flat.device)
            context_indices = indices[:num_context_eval]
            target_indices = indices[num_context_eval : num_context_eval + num_target_eval]

            x_context = x_for_split[:, context_indices, :]
            y_context_labels = y_for_split[:, context_indices]
            x_target = x_for_split[:, target_indices, :]
            y_target_labels = y_for_split[:, target_indices]
            
            if num_context_eval == 0: # Use a generic prior or skip if model requires context
                 # Fallback: use the task's learned prior mean for z directly. This is a simplification.
                task_prior_m, _ = model.get_task_prior(eval_task_id)
                z_sample = task_prior_m.unsqueeze(0) # batch_size 1
                if x_target.size(1) > 0: # if there are targets
                    y_pred_logits = model.decoder(z_sample, x_target)
                else: continue # no targets to predict
            else:
                y_context_onehot = F.one_hot(y_context_labels, num_classes=model.y_dim_onehot).float()
                y_pred_logits, _, _ = model(x_context, y_context_onehot, x_target)

            _, predicted = torch.max(y_pred_logits.data, -1)
            total_correct += (predicted == y_target_labels).sum().item()
            total_targets += y_target_labels.numel()
            
    accuracy = 100. * total_correct / total_targets if total_targets > 0 else 0
    return accuracy

# --- Run Continual Learning Experiment ---
def run_continual_learning_np_dynamic_prior_experiment():
    x_dim, y_dim_onehot, r_dim, z_dim = 784, 10, 128, 64 # smaller r_dim, z_dim for speed
    enc_hidden_dim, dec_hidden_dim, y_dim_out = 128, 128, 10

    model = NeuralProcessDynamicPrior(NUM_TASKS, x_dim, y_dim_onehot, r_dim, z_dim, 
                                      enc_hidden_dim, dec_hidden_dim, y_dim_out).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4) # Adjusted LR
    
    epochs_per_task = 10 # Reduced for speed
    batch_size_train = 32 # Reduced for stability with splitting
    kl_weight_train = 0.01

    task_accuracies = {i: [] for i in range(NUM_TASKS)}
    avg_accuracies_over_time = []

    for current_task_idx in range(NUM_TASKS):
        print(f"\n--- Training on Task {current_task_idx+1}/{NUM_TASKS} ---")
        train_np_task(model, optimizer, current_task_idx, epochs=epochs_per_task, 
                      batch_size=batch_size_train, kl_weight=kl_weight_train)
        
        current_eval_accuracies = []
        print(f"\n--- Evaluating after Task {current_task_idx+1} ---")
        for eval_task_idx in range(NUM_TASKS):
            if eval_task_idx <= current_task_idx:
                acc = evaluate_np_task(model, eval_task_idx, num_eval_batches=30, 
                                       batch_size=32, fixed_num_context=15) # reduced eval params
                task_accuracies[eval_task_idx].append(acc)
                current_eval_accuracies.append(acc)
                print(f"  Accuracy on Task {eval_task_idx+1}: {acc:.2f}%")
            else:
                task_accuracies[eval_task_idx].append(np.nan)
        
        if current_eval_accuracies:
             avg_acc = np.mean([acc for acc in current_eval_accuracies if not np.isnan(acc)])
             avg_accuracies_over_time.append(avg_acc)
             print(f"  Average accuracy over seen tasks: {avg_acc:.2f}%")

    return task_accuracies, avg_accuracies_over_time, model

# --- Run and Plot ---
print("Starting Neural Process (Dynamic Prior) Continual Learning experiment...")
task_accuracies_np_dp, avg_accuracies_np_dp, trained_model_dp = run_continual_learning_np_dynamic_prior_experiment()

# Plotting (same as before, adapted for new variable names)
plt.figure(figsize=(12, 7))
for task_id, accuracies in task_accuracies_np_dp.items():
    stages = list(range(1, NUM_TASKS + 1))
    plot_accuracies = [np.nan] * task_id + accuracies[:NUM_TASKS - task_id]
    if len(plot_accuracies) < NUM_TASKS:
        plot_accuracies.extend([np.nan] * (NUM_TASKS - len(plot_accuracies)))
    plt.plot(stages, plot_accuracies[:NUM_TASKS], marker='o', linestyle='-', label=f'Task {task_id+1} Perf.')

plt.plot(range(1, NUM_TASKS + 1), avg_accuracies_np_dp, marker='x', linestyle='--', color='black', label='Avg. Accuracy (Seen Tasks)')
plt.xlabel('Training Stage (After Training on Task X)')
plt.ylabel('Accuracy (%)')
plt.title(f'NP Dynamic Priors ({NUM_TASKS} Tasks): Continual Learning on Permuted MNIST')
plt.xticks(range(1, NUM_TASKS + 1))
plt.legend(loc='best')
plt.grid(True)
plt.ylim(0, 101)
plt.savefig(f'np_dynamic_priors_forgetting_{NUM_TASKS}tasks.png')
plt.show()

print("\n--- Final Accuracies Table (Dynamic Priors) ---")
header = "Trained on Task -> |"
for i in range(NUM_TASKS): header += f"   Task {i+1}   |"
print(header); print("-" * len(header))
for eval_task_id in range(NUM_TASKS):
    row_str = f"Perf. on Task {eval_task_id+1:2d}  | "
    for train_stage_idx in range(NUM_TASKS):
        if train_stage_idx < eval_task_id: row_str += "    -     | "
        else:
            acc_idx = train_stage_idx - eval_task_id
            if acc_idx < len(task_accuracies_np_dp[eval_task_id]):
                 acc_val = task_accuracies_np_dp[eval_task_id][acc_idx]
                 row_str += f"{acc_val:6.2f}   | " if not np.isnan(acc_val) else "  N/A   | "
            else: row_str += " Error  | "
    print(row_str)

if avg_accuracies_np_dp: print(f"\nFinal avg accuracy (all tasks trained): {avg_accuracies_np_dp[-1]:.2f}%")

# Optional: Inspect learned prior means
print("\nLearned Task Prior Means (first 5 dims):")
for i in range(NUM_TASKS):
    print(f"Task {i+1} Prior Mean: {trained_model_dp.task_prior_means[i,:5].detach().cpu().numpy()}")
    # print(f"Task {i+1} Prior LogVar: {trained_model_dp.task_prior_logvars[i,:5].detach().cpu().numpy()}")