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
# and move processing to CPU initially to avoid OOM on GPU for large datasets
x_train_np = mnist_train_full.data.view(-1, 28*28).float() / 255.0
y_train_np = mnist_train_full.targets
x_test_np = mnist_test_full.data.view(-1, 28*28).float() / 255.0
y_test_np = mnist_test_full.targets

num_tasks = 10 # Reduced for quicker demonstration; can be increased
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

# --- Neural Process Model ---
class NPEncoder(nn.Module):
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

class LatentEncoder(nn.Module):
    """Takes aggregated r_mean and maps it to latent z parameters."""
    def __init__(self, r_dim, z_dim):
        super(LatentEncoder, self).__init__()
        self.fc_mean = nn.Linear(r_dim, z_dim)
        self.fc_logvar = nn.Linear(r_dim, z_dim)

    def forward(self, r_aggregated):
        # r_aggregated: (batch_size, r_dim)
        z_mean = self.fc_mean(r_aggregated)
        z_logvar = self.fc_logvar(r_aggregated)
        return z_mean, z_logvar

class NPDecoder(nn.Module):
    """Decodes (z, x_target) into y_target predictions."""
    def __init__(self, x_dim, z_dim, hidden_dim, y_dim_out):
        super(NPDecoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(z_dim + x_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, y_dim_out) # Outputting logits for classification
        )

    def forward(self, z_sample, x_target):
        # z_sample: (batch_size, z_dim)
        # x_target: (batch_size, num_target_points, x_dim)
        
        num_target_points = x_target.size(1)
        # Repeat z for each target point
        z_repeated = z_sample.unsqueeze(1).repeat(1, num_target_points, 1) # (batch, num_target, z_dim)
        
        zx_cat = torch.cat([z_repeated, x_target], dim=-1)
        return self.fc(zx_cat)


class NeuralProcess(nn.Module):
    def __init__(self, x_dim=784, y_dim_onehot=10, r_dim=128, z_dim=64, enc_hidden_dim=128, dec_hidden_dim=128, y_dim_out=10):
        super(NeuralProcess, self).__init__()
        self.x_dim = x_dim
        self.y_dim_onehot = y_dim_onehot
        self.y_dim_out = y_dim_out

        self.xy_encoder = NPEncoder(x_dim, y_dim_onehot, enc_hidden_dim, r_dim)
        self.latent_encoder = LatentEncoder(r_dim, z_dim)
        self.decoder = NPDecoder(x_dim, z_dim, dec_hidden_dim, y_dim_out)

    def aggregate(self, r_i):
        # r_i: (batch_size, num_context_points, r_dim)
        return torch.mean(r_i, dim=1) # (batch_size, r_dim)

    def reparameterize(self, z_mean, z_logvar):
        std = torch.exp(0.5 * z_logvar)
        eps = torch.randn_like(std)
        return z_mean + eps * std

    def forward(self, x_context, y_context_onehot, x_target, y_target_onehot=None):
        # x_context: (batch_size, num_context, x_dim)
        # y_context_onehot: (batch_size, num_context, y_dim_onehot)
        # x_target: (batch_size, num_target, x_dim)
        # y_target_onehot: (batch_size, num_target, y_dim_onehot) - for KL calculation if needed, not directly used by decoder for prediction output

        r_i_context = self.xy_encoder(x_context, y_context_onehot) # (batch, num_context, r_dim)
        r_aggregated = self.aggregate(r_i_context) # (batch, r_dim)
        
        z_mean, z_logvar = self.latent_encoder(r_aggregated) # (batch, z_dim)
        
        # For training, we sample z. For evaluation, we might use the mean.
        # Here we always sample for simplicity in this baseline.
        z_sample = self.reparameterize(z_mean, z_logvar) # (batch, z_dim)
        
        y_pred_logits = self.decoder(z_sample, x_target) # (batch, num_target, y_dim_out)
        
        return y_pred_logits, z_mean, z_logvar

# --- Loss Function ---
def np_loss(y_pred_logits, y_target_labels, z_mean, z_logvar, kl_weight=0.1):
    # y_pred_logits: (batch, num_target, y_dim_out)
    # y_target_labels: (batch, num_target) - class indices
    
    # Reconstruction Loss (Cross-Entropy)
    # Reshape for cross_entropy: (batch * num_target, y_dim_out) and (batch * num_target)
    ce_loss = F.cross_entropy(y_pred_logits.view(-1, y_pred_logits.size(-1)), 
                              y_target_labels.view(-1), reduction='mean')
    
    # KL Divergence
    # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_div = -0.5 * torch.sum(1 + z_logvar - z_mean.pow(2) - z_logvar.exp(), dim=1)
    kl_loss = torch.mean(kl_div)
    
    total_loss = ce_loss + kl_weight * kl_loss
    return total_loss, ce_loss, kl_loss

# --- Training and Evaluation Helper ---
def get_context_target_split(x_batch, y_batch, num_context_range=(5, 50), num_extra_target_range=(10,100)):
    batch_size, total_points, _ = x_batch.shape
    
    num_context = np.random.randint(num_context_range[0], min(num_context_range[1] + 1, total_points))
    
    # Ensure we have enough points for at least one target if all context is taken
    max_target_points = total_points - num_context
    if max_target_points <=0: # if num_context takes all points
        num_context = total_points -1 # leave at least one for target
        if num_context <1 and total_points > 0: # if only one point total
            num_context = 0 # then no context, all target (special case handled by model if num_context=0)
        elif total_points == 0:
            return None, None, None, None, None, None # Should not happen with dataloader

    num_target = np.random.randint(num_extra_target_range[0], min(num_extra_target_range[1] + 1, max_target_points +1 ))
    if num_target == 0 and max_target_points > 0 : num_target = 1


    indices = torch.randperm(total_points)
    context_indices = indices[:num_context]
    target_indices = indices[num_context : num_context + num_target]

    if num_context == 0: # No context points, NP might struggle or this case needs special handling.
                        # For simplicity, we ensure at least a few context points typically.
                        # Or, the latent encoder uses a fixed prior if r_aggregated is empty.
                        # Here, we'll just make sure num_context is at least 1 if possible.
        if total_points > 1:
            num_context = 1
            context_indices = indices[:num_context]
            target_indices = indices[num_context : num_context + num_target] if num_context + num_target <= total_points else indices[num_context:]
        elif total_points == 1: # only one point
            # use it as target, no context
            context_indices = torch.empty(0, dtype=torch.long)
            target_indices = indices
            num_context = 0
            num_target = 1


    x_context = x_batch[:, context_indices, :]
    y_context_labels = y_batch[:, context_indices]
    
    x_target = x_batch[:, target_indices, :]
    y_target_labels = y_batch[:, target_indices]

    # One-hot encode y_context
    y_context_onehot = F.one_hot(y_context_labels, num_classes=10).float()
    
    return x_context, y_context_onehot, y_context_labels, x_target, y_target_labels, num_context > 0


def train_np_task(model, optimizer, task_id, epochs=5, batch_size=64, kl_weight=0.01):
    x_train_task_np, y_train_task_np = task_data_train[task_id]
    
    # Create a dataset of (image, label) tuples for this task
    # The NP expects variable number of context/target points, so we process this in the loop
    # However, DataLoader can help batch original images
    train_dataset = TensorDataset(x_train_task_np, y_train_task_np)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    model.train()
    print(f"Training on Task {task_id+1}...")
    for epoch in range(epochs):
        total_loss_epoch = 0
        total_ce_epoch = 0
        total_kl_epoch = 0
        correct_preds = 0
        total_target_points = 0
        
        for batch_idx, (x_batch_flat, y_batch_labels_flat) in enumerate(train_loader):
            # x_batch_flat: (batch_size, 784), y_batch_labels_flat: (batch_size)
            x_batch_flat, y_batch_labels_flat = x_batch_flat.to(device), y_batch_labels_flat.to(device)

            # Reshape to (batch_size, num_points=1 initially, dim) as NP expects num_points dimension
            # For NP, each item in batch is a "set" of points. Here, each image is a set of 1 point.
            # We will combine images in a batch to form a larger "meta-batch" for context/target split.
            # This means we take all batch_size * 1 points and split them.
            
            current_batch_size = x_batch_flat.size(0)
            x_for_split = x_batch_flat.view(1, current_batch_size, -1) # (1, current_batch_size, 784)
            y_for_split = y_batch_labels_flat.view(1, current_batch_size)   # (1, current_batch_size)


            x_context, y_context_onehot, _, x_target, y_target_labels, has_context = \
                get_context_target_split(x_for_split, y_for_split, 
                                         num_context_range=(max(1, int(current_batch_size * 0.1)), int(current_batch_size * 0.7)), # min 1 context point
                                         num_extra_target_range=(max(1, int(current_batch_size * 0.2)), int(current_batch_size * 0.9)))


            if not has_context or x_target.size(1) == 0: # Skip if no context or no target points
                continue

            optimizer.zero_grad()
            y_pred_logits, z_mean, z_logvar = model(x_context, y_context_onehot, x_target)
            
            loss, ce, kl = np_loss(y_pred_logits, y_target_labels, z_mean, z_logvar, kl_weight=kl_weight)
            
            loss.backward()
            optimizer.step()
            
            total_loss_epoch += loss.item()
            total_ce_epoch += ce.item()
            total_kl_epoch += kl.item()
            
            _, predicted_labels = torch.max(y_pred_logits.data, -1)
            correct_preds += (predicted_labels == y_target_labels).sum().item()
            total_target_points += y_target_labels.numel()

        avg_loss = total_loss_epoch / (batch_idx +1)
        avg_ce = total_ce_epoch / (batch_idx +1)
        avg_kl = total_kl_epoch / (batch_idx +1)
        accuracy = 100. * correct_preds / total_target_points if total_target_points > 0 else 0
        print(f'  Epoch {epoch+1}/{epochs}: Avg Loss: {avg_loss:.4f} (CE: {avg_ce:.4f}, KL: {avg_kl:.4f}), Acc: {accuracy:.2f}%')


def evaluate_np_task(model, task_id, num_eval_batches=50, batch_size=64, fixed_num_context=20):
    x_test_task_np, y_test_task_np = task_data_test[task_id]
    test_dataset = TensorDataset(x_test_task_np, y_test_task_np)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True) # Shuffle to get diverse context/targets

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
            x_for_split = x_batch_flat.view(1, current_batch_size, -1) 
            y_for_split = y_batch_labels_flat.view(1, current_batch_size)

            # For evaluation, use a fixed number of context points if available
            num_context_eval = min(fixed_num_context, current_batch_size -1)
            if num_context_eval < 1 and current_batch_size > 1: num_context_eval = 1
            if current_batch_size <= 1: num_context_eval = 0 # no context if only 1 point

            num_target_eval = current_batch_size - num_context_eval
            if num_target_eval <=0 : continue # need targets to evaluate

            indices = torch.randperm(current_batch_size)
            context_indices = indices[:num_context_eval]
            target_indices = indices[num_context_eval : num_context_eval + num_target_eval]

            x_context = x_for_split[:, context_indices, :]
            y_context_labels = y_for_split[:, context_indices]
            x_target = x_for_split[:, target_indices, :]
            y_target_labels = y_for_split[:, target_indices]
            
            if num_context_eval == 0: # If no context, use a prior for z (mean=0, logvar=0)
                # This requires slight modification in model or a specific path.
                # For simplicity, we ensure some context if possible, or skip.
                # A more robust NP handles zero context by falling back to prior for z.
                # Here, we'll just use the model as is, it might use an empty context if x_context is empty.
                # Let's ensure y_context_onehot is correctly shaped even if empty.
                y_context_onehot = torch.empty(1, 0, model.y_dim_onehot, device=device)
                if x_context.nelement() == 0: # literally no context points
                     # Create dummy z_mean, z_logvar based on prior
                    z_mean = torch.zeros(1, model.latent_encoder.fc_mean.out_features, device=device)
                    z_logvar = torch.zeros(1, model.latent_encoder.fc_logvar.out_features, device=device) # log(1) = 0 for variance 1
                    z_sample = model.reparameterize(z_mean, z_logvar)
                    y_pred_logits = model.decoder(z_sample, x_target)
                else: # this should not be hit if num_context_eval is 0
                    y_context_onehot = F.one_hot(y_context_labels, num_classes=model.y_dim_onehot).float()
                    y_pred_logits, _, _ = model(x_context, y_context_onehot, x_target)

            else:
                y_context_onehot = F.one_hot(y_context_labels, num_classes=model.y_dim_onehot).float()
                y_pred_logits, _, _ = model(x_context, y_context_onehot, x_target) # z_mean, z_logvar not used for acc

            _, predicted = torch.max(y_pred_logits.data, -1)
            total_correct += (predicted == y_target_labels).sum().item()
            total_targets += y_target_labels.numel()
            
    accuracy = 100. * total_correct / total_targets if total_targets > 0 else 0
    return accuracy


# --- Run Continual Learning Experiment ---
def run_continual_learning_np_experiment():
    # NP Hyperparameters
    x_dim = 784
    y_dim_onehot = 10 
    r_dim = 256       # Representation dim for each (x,y) pair
    z_dim = 128       # Latent dim
    enc_hidden_dim = 256
    dec_hidden_dim = 256
    y_dim_out = 10    # Output logits for 10 classes

    model = NeuralProcess(x_dim, y_dim_onehot, r_dim, z_dim, enc_hidden_dim, dec_hidden_dim, y_dim_out).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4) # NPs can be sensitive to LR
    
    epochs_per_task = 10 # Increased epochs as NP training can be slower
    batch_size_train = 32  # Smaller batch for NP context/target splits within batch
    kl_weight_train = 0.01 # Weight for KL term in loss

    task_accuracies = {i: [] for i in range(num_tasks)}
    avg_accuracies_over_time = []


    for current_task_idx in range(num_tasks):
        print(f"\n--- Training on Task {current_task_idx+1}/{num_tasks} ---")
        train_np_task(model, optimizer, current_task_idx, epochs=epochs_per_task, batch_size=batch_size_train, kl_weight=kl_weight_train)
        
        current_eval_accuracies = []
        print(f"\n--- Evaluating after Task {current_task_idx+1} ---")
        for eval_task_idx in range(num_tasks):
            if eval_task_idx <= current_task_idx:
                acc = evaluate_np_task(model, eval_task_idx, num_eval_batches=30, batch_size=64, fixed_num_context=30)
                task_accuracies[eval_task_idx].append(acc)
                current_eval_accuracies.append(acc)
                print(f"  Accuracy on Task {eval_task_idx+1}: {acc:.2f}%")
            else:
                # For tasks not yet seen, append NaN or a placeholder if plotting all tasks from start
                task_accuracies[eval_task_idx].append(np.nan) # Will not be plotted until seen
        
        if current_eval_accuracies: # only average over seen tasks
             avg_acc = np.mean([acc for acc in current_eval_accuracies if not np.isnan(acc)])
             avg_accuracies_over_time.append(avg_acc)
             print(f"  Average accuracy over seen tasks: {avg_acc:.2f}%")


    return task_accuracies, avg_accuracies_over_time

# --- Run and Plot ---
print("Starting Neural Process Continual Learning experiment...")
task_accuracies_np, avg_accuracies_np = run_continual_learning_np_experiment()

# Plotting the results
plt.figure(figsize=(12, 7))
for task_id, accuracies in task_accuracies_np.items():
    # Create x-values for plotting: starts from task_id
    # Each accuracy entry corresponds to a training stage (after training task 0, 1, ...)
    stages = list(range(1, num_tasks + 1))
    
    # Pad with NaNs at the beginning for tasks not yet introduced
    plot_accuracies = [np.nan] * task_id + accuracies[:num_tasks - task_id]
    # Ensure it's the correct length for all stages
    if len(plot_accuracies) < num_tasks:
        plot_accuracies.extend([np.nan] * (num_tasks - len(plot_accuracies)))
    
    plt.plot(stages, plot_accuracies[:num_tasks], marker='o', linestyle='-', label=f'Task {task_id+1} Perf.')

plt.plot(range(1, num_tasks + 1), avg_accuracies_np, marker='x', linestyle='--', color='black', label='Avg. Accuracy (Seen Tasks)')
plt.xlabel('Training Stage (After Training on Task X)')
plt.ylabel('Accuracy (%)')
plt.title('Neural Process: Continual Learning on Permuted MNIST')
plt.xticks(range(1, num_tasks + 1))
plt.legend(loc='best')
plt.grid(True)
plt.ylim(0, 101)
plt.savefig('np_catastrophic_forgetting.png')
plt.show()

print("\n--- Final Accuracies Table (Performance on Task Y after training on Task X) ---")
# Header
header = "Trained on Task -> |"
for i in range(num_tasks):
    header += f"   Task {i+1}   |"
print(header)
print("-" * len(header))

# Rows: Performance of each task
for eval_task_id in range(num_tasks):
    row_str = f"Perf. on Task {eval_task_id+1:2d}  | "
    for train_stage_idx in range(num_tasks):
        if train_stage_idx < eval_task_id:
            row_str += "    -     | " # Task not yet introduced for training this eval_task
        else:
            # task_accuracies[eval_task_id] stores accuracies AFTER each training stage
            # The index into this list corresponds to (training_stage_idx - eval_task_id)
            acc_idx = train_stage_idx - eval_task_id
            if acc_idx < len(task_accuracies_np[eval_task_id]):
                 acc_val = task_accuracies_np[eval_task_id][acc_idx]
                 row_str += f"{acc_val:6.2f}   | " if not np.isnan(acc_val) else "  N/A   | "
            else: # Should not happen if logic correct
                 row_str += " Error  | "
    print(row_str)

print("\nFinal average accuracy across tasks seen so far, after all training:")
if avg_accuracies_np:
    print(f"{avg_accuracies_np[-1]:.2f}%")