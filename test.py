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

from torch.func import functional_call, jacrev

torch.manual_seed(42)
np.random.seed(42)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- OPM Hyperparameters ---
M_JACOBIAN_SAMPLES = 100 # Number of x_val samples for Jacobian per task
NUM_CONTEXT_JACOBIAN = 100 # Number of context points for C_i for Jacobian
JACOBIAN_PROJ_REG = 1e-5 # Regularization for pinv in projection


# --- Data Loading and Permutation ---
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
mnist_train_full = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
mnist_test_full = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

x_train_np = mnist_train_full.data.view(-1, 28*28).float() / 255.0
y_train_np = mnist_train_full.targets
x_test_np = mnist_test_full.data.view(-1, 28*28).float() / 255.0
y_test_np = mnist_test_full.targets

num_tasks = 3 # Reduced for quicker demonstration; can be increased
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

# --- Neural Process Model Components ---
class NPEncoder(nn.Module):
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
        xy_cat = torch.cat([x, y], dim=-1)
        return self.fc(xy_cat)

class LatentEncoder(nn.Module):
    def __init__(self, r_dim, z_dim):
        super(LatentEncoder, self).__init__()
        self.fc_mean = nn.Linear(r_dim, z_dim)
        self.fc_logvar = nn.Linear(r_dim, z_dim)
    def forward(self, r_aggregated):
        z_mean = self.fc_mean(r_aggregated)
        z_logvar = self.fc_logvar(r_aggregated)
        return z_mean, z_logvar

class NPDecoder(nn.Module):
    def __init__(self, x_dim, z_dim, hidden_dim, y_dim_out):
        super(NPDecoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(z_dim + x_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, y_dim_out)
        )
    def forward(self, z_sample, x_target):
        num_target_points = x_target.size(1)
        z_repeated = z_sample.unsqueeze(1).repeat(1, num_target_points, 1)
        zx_cat = torch.cat([z_repeated, x_target], dim=-1)
        return self.fc(zx_cat)

class NeuralProcessOPM(nn.Module):
    def __init__(self, x_dim=784, y_dim_onehot=10, r_dim=128, z_dim=64,
                 enc_hidden_dim=128, dec_hidden_dim=128, y_dim_out=10):
        super(NeuralProcessOPM, self).__init__()
        self.x_dim = x_dim
        self.y_dim_onehot = y_dim_onehot
        self.y_dim_out = y_dim_out
        self.r_dim = r_dim
        self.z_dim = z_dim
        self.enc_hidden_dim = enc_hidden_dim
        self.dec_hidden_dim = dec_hidden_dim

        self.xy_encoder = NPEncoder(x_dim, y_dim_onehot, enc_hidden_dim, r_dim)
        self.latent_encoder = LatentEncoder(r_dim, z_dim)

        self.phi_modules = nn.ModuleList([self.xy_encoder, self.latent_encoder])
        self.decoders = nn.ModuleList()
        self.current_task_idx_internal = 0
        self.past_task_jacobians_stacked = None

    def get_phi_parameters(self):
        params = []
        for module in self.phi_modules:
            params.extend(list(module.parameters()))
        return params

    def add_decoder_for_task(self):
        new_decoder = NPDecoder(self.x_dim, self.z_dim, self.dec_hidden_dim, self.y_dim_out).to(device)
        self.decoders.append(new_decoder)
        print(f"Added decoder for task {len(self.decoders)-1}. Total decoders: {len(self.decoders)}")

    def aggregate(self, r_i):
        if r_i.size(1) == 0:
            batch_size = r_i.size(0)
            return torch.zeros(batch_size, self.r_dim, device=r_i.device)
        return torch.mean(r_i, dim=1)

    def reparameterize(self, z_mean, z_logvar):
        std = torch.exp(0.5 * z_logvar)
        eps = torch.randn_like(std)
        return z_mean + eps * std

    def forward(self, x_context, y_context_onehot, x_target, task_id_for_decoder=None):
        r_i_context = self.xy_encoder(x_context, y_context_onehot)
        r_aggregated = self.aggregate(r_i_context)
        z_mean, z_logvar = self.latent_encoder(r_aggregated)
        z_sample = self.reparameterize(z_mean, z_logvar)

        dec_idx = task_id_for_decoder if task_id_for_decoder is not None else self.current_task_idx_internal
        if dec_idx >= len(self.decoders):
            raise ValueError(f"Decoder for task {dec_idx} not found. Available: {len(self.decoders)}")

        selected_decoder = self.decoders[dec_idx]
        y_pred_logits = selected_decoder(z_sample, x_target)
        return y_pred_logits, z_mean, z_logvar

# --- Loss Function ---
def np_loss(y_pred_logits, y_target_labels, z_mean, z_logvar, kl_weight=0.1):
    ce_loss = F.cross_entropy(y_pred_logits.view(-1, y_pred_logits.size(-1)),
                              y_target_labels.view(-1), reduction='mean')
    kl_div = -0.5 * torch.sum(1 + z_logvar - z_mean.pow(2) - z_logvar.exp(), dim=1)
    kl_loss = torch.mean(kl_div)
    total_loss = ce_loss + kl_weight * kl_loss
    return total_loss, ce_loss, kl_loss

# --- Training and Evaluation Helper ---
def get_context_target_split(x_batch, y_batch, num_context_range=(5, 50), num_extra_target_range=(10,100), y_dim_onehot=10):
    batch_size, total_points, _ = x_batch.shape
    min_ctx = num_context_range[0]
    max_ctx = min(num_context_range[1] + 1, total_points -1)
    if min_ctx >= max_ctx : min_ctx = max(0,max_ctx-1)
    num_context = 0
    if total_points > 1 and min_ctx < max_ctx :
        num_context = np.random.randint(min_ctx, max_ctx)
    elif total_points > 1 and min_ctx == max_ctx and min_ctx > 0 :
        num_context = min_ctx
    elif total_points == 1:
        num_context = 0
    max_target_points = total_points - num_context
    min_trg = num_extra_target_range[0]
    max_trg = min(num_extra_target_range[1] + 1, max_target_points +1)
    if min_trg >= max_trg: min_trg = max(1, max_trg-1)
    num_target = 0
    if max_target_points > 0 and min_trg < max_trg:
        num_target = np.random.randint(min_trg, max_trg)
    elif max_target_points > 0 and min_trg == max_trg and min_trg >0:
        num_target = min_trg
    if total_points == 0: return None, None, None, None, None, False
    if num_context == 0 and num_target == 0 and total_points > 0: num_target = total_points
    indices = torch.randperm(total_points, device=x_batch.device)
    context_indices = indices[:num_context]
    target_indices = indices[num_context : num_context + num_target]
    x_context = x_batch[:, context_indices, :]
    y_context_labels = y_batch[:, context_indices]
    x_target = x_batch[:, target_indices, :]
    y_target_labels = y_batch[:, target_indices]
    y_context_onehot = F.one_hot(y_context_labels.long(), num_classes=y_dim_onehot).float()
    return x_context, y_context_onehot, y_context_labels, x_target, y_target_labels, num_context > 0

# --- OPM Specific Functions ---
@torch.no_grad()
def get_opm_data_for_jacobian(task_id, num_ctx, num_val_targets, y_dim_onehot):
    x_task_full_np, y_task_full_np = task_data_train[task_id]
    total_samples = x_task_full_np.size(0)
    if total_samples < num_ctx + num_val_targets:
        # print(f"Warning: Not enough samples in task {task_id} for Jacobian data. Required {num_ctx + num_val_targets}, have {total_samples}")
        num_ctx = min(num_ctx, max(0, total_samples - num_val_targets))
        num_val_targets = min(num_val_targets, total_samples - num_ctx)
        if num_val_targets == 0 and total_samples > num_ctx : num_val_targets = total_samples - num_ctx
        if num_ctx == 0 and total_samples > 0 and num_val_targets < total_samples : num_ctx = total_samples - num_val_targets
    if num_ctx == 0 and num_val_targets == 0: return None, None, None
    indices = torch.randperm(total_samples)
    ctx_indices = indices[:num_ctx]
    val_target_indices = indices[num_ctx : num_ctx + num_val_targets]
    context_x_jac = x_task_full_np[ctx_indices].to(device).unsqueeze(0)
    context_y_labels_jac = y_task_full_np[ctx_indices].to(device).unsqueeze(0)
    context_y_onehot_jac = F.one_hot(context_y_labels_jac.long(), num_classes=y_dim_onehot).float()
    target_x_val_jac = x_task_full_np[val_target_indices].to(device).unsqueeze(0)
    if target_x_val_jac.size(1) == 0: return None, None, None
    return context_x_jac, context_y_onehot_jac, target_x_val_jac

# MODIFIED FUNCTION USING torch.func
def collect_jacobian_for_task_opm_func(model, task_id_of_decoder, context_x_task_i, context_y_onehot_task_i, target_x_val_task_i):
    """Computes Jacobian matrix J_i = ∂(D_θ_i(E_φ(C_i), X_val))/∂φ using torch.func."""
    # model.eval() # Already done before calling this in the main loop typically
    device_ = next(model.parameters()).device

    context_x_task_i = context_x_task_i.to(device_)
    context_y_onehot_task_i = context_y_onehot_task_i.to(device_)
    target_x_val_task_i = target_x_val_task_i.to(device_)

    phi_params_list = []
    phi_param_names_ordered_xy = []
    phi_param_names_ordered_le = []

    for name, p in model.xy_encoder.named_parameters():
        phi_params_list.append(p.detach().clone())
        phi_param_names_ordered_xy.append(name)
    for name, p in model.latent_encoder.named_parameters():
        phi_params_list.append(p.detach().clone())
        phi_param_names_ordered_le.append(name)
    phi_params_tuple = tuple(phi_params_list)

    phi_buffers_xy = {name: b.detach().clone() for name, b in model.xy_encoder.named_buffers()}
    phi_buffers_le = {name: b.detach().clone() for name, b in model.latent_encoder.named_buffers()}

    current_decoder = model.decoders[task_id_of_decoder]

    def compute_outputs_functional(phi_params_runtime, context_x, context_y_onehot, target_x_val):
        num_params_xy_encoder = len(phi_param_names_ordered_xy)

        xy_encoder_params_runtime_tuple = phi_params_runtime[:num_params_xy_encoder]
        latent_encoder_params_runtime_tuple = phi_params_runtime[num_params_xy_encoder:]

        xy_param_dict = {name: param for name, param in zip(phi_param_names_ordered_xy, xy_encoder_params_runtime_tuple)}
        le_param_dict = {name: param for name, param in zip(phi_param_names_ordered_le, latent_encoder_params_runtime_tuple)}

        r_i_context_jac = functional_call(
            model.xy_encoder,
            (xy_param_dict, phi_buffers_xy),
            args=(context_x, context_y_onehot)
        )
        r_aggregated_jac = model.aggregate(r_i_context_jac)
        z_mean_jac, _ = functional_call(
            model.latent_encoder,
            (le_param_dict, phi_buffers_le),
            args=(r_aggregated_jac,)
        )
        y_pred_logits_jac = current_decoder(z_mean_jac, target_x_val)
        return y_pred_logits_jac.flatten()

    J_i_tuple_of_tensors = jacrev(compute_outputs_functional, argnums=0)(
        phi_params_tuple, context_x_task_i, context_y_onehot_task_i, target_x_val_task_i
    )

    J_i_flat_list = []
    if not J_i_tuple_of_tensors: # Should not happen if phi_params_tuple is not empty
        print("Warning: jacrev returned empty result.")
        return torch.empty(0, 0, device=device_)

    for J_p in J_i_tuple_of_tensors:
        if J_p is None: # Can happen if a parameter is not part of the computation graph to the output
            print("Warning: jacrev returned None for a parameter's Jacobian. Filling with zeros.")
            # This requires knowing the shape of the parameter this None corresponds to.
            # For simplicity, we'll skip this, but ideally, it should be handled or indicate an error.
            # Find corresponding param in phi_params_tuple to get its numel to fill with zeros
            # This part is tricky as we don't have direct index correspondence here
            # J_i_flat_list.append(torch.zeros(y_pred_logits_jac_flat_size, param_numel, device=device_))
            continue # Or raise error
        J_i_flat_list.append(J_p.reshape(J_p.shape[0], -1))

    if not J_i_flat_list:
        print("Warning: No valid Jacobians generated after processing jacrev output.")
        return torch.empty(0, 0, device=device_)

    J_i = torch.cat(J_i_flat_list, dim=1)
    return J_i.detach()


def project_gradients_opm(model, reg=1e-5, proj_dim=100):
    if model.past_task_jacobians_stacked is None or model.past_task_jacobians_stacked.numel() == 0:
        return

    J_old = model.past_task_jacobians_stacked
    device_ = J_old.device

    phi_params_for_grad = model.get_phi_parameters()
    g_list = []
    valid_phi_params_for_grad = [] # Keep track of params that actually have grads

    for p in phi_params_for_grad:
        if p.grad is not None:
            g_list.append(p.grad.flatten())
            valid_phi_params_for_grad.append(p)
        # else:
            # If a parameter is part of phi_modules but doesn't have a gradient (e.g., frozen),
            # it shouldn't contribute to 'g'. J_old should ideally correspond only to
            # parameters that *are* being trained and have gradients.
            # print(f"Warning: Param {p.shape} in phi_modules has no grad during projection.")

    if not g_list:
        # print("No gradients in phi_parameters for OPM projection.")
        return

    g = torch.cat(g_list).to(device_)

    # Crucial: J_old (num_constraints, phi_dim) must have phi_dim matching g.shape[0]
    # This means the parameters included when J_old was created must be the same (and in same order)
    # as those contributing to 'g'.
    if J_old.shape[1] != g.shape[0]:
        print(f"CRITICAL WARNING: Mismatch in OPM projection dimensions. J_old.shape[1]={J_old.shape[1]}, g.shape[0]={g.shape[0]}. "
              f"This usually means the set of parameters used for Jacobian calculation differs from those "
              f"currently having gradients. Skipping projection. This needs to be fixed.")
        return

    J_old_effective = J_old
    if J_old.size(0) > proj_dim and proj_dim > 0 : # If number of constraints > proj_dim, then project J_old's rows
        # print(f"Projecting J_old from {J_old.size(0)} to {proj_dim} constraints using random projection.")
        rand_proj_matrix = torch.randn(J_old.size(0), proj_dim, device=device_)
        rand_proj_matrix = rand_proj_matrix / (torch.norm(rand_proj_matrix, dim=0, keepdim=True) + 1e-9) # Normalize columns
        J_old_effective = rand_proj_matrix.T @ J_old # (proj_dim, phi_dim)

    A = J_old_effective @ J_old_effective.T + reg * torch.eye(J_old_effective.size(0), device=device_)
    B = J_old_effective @ g

    try:
        x = torch.linalg.solve(A, B)
        g_proj_flat = g - J_old_effective.T @ x
    except torch.linalg.LinAlgError as e:
        print(f"LinAlgError during OPM projection: {e}. Attempting pseudo-inverse.")
        try:
            pinv_A = torch.linalg.pinv(A)
            x = pinv_A @ B
            g_proj_flat = g - J_old_effective.T @ x
            print("  Used pseudo-inverse for projection solve.")
        except torch.linalg.LinAlgError as e_pinv:
            print(f"  LinAlgError with pseudo-inverse as well: {e_pinv}. Skipping OPM projection for this step.")
            g_proj_flat = g # No projection

    offset = 0
    for p in valid_phi_params_for_grad: # Use the list of params that contributed to g
        numel = p.numel()
        if p.grad is not None: # Should be true by construction of valid_phi_params_for_grad
            p.grad.data = g_proj_flat[offset:offset+numel].view_as(p.grad.data)
        offset += numel


def train_np_task(model, current_task_idx, optimizer_phi, optimizer_decoder, epochs=5, batch_size=64, kl_weight=0.01):
    x_train_task_np, y_train_task_np = task_data_train[current_task_idx]
    train_dataset = TensorDataset(x_train_task_np, y_train_task_np)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    model.train()
    for i, dec in enumerate(model.decoders):
        for param in dec.parameters():
            param.requires_grad = (i == current_task_idx)
    for param in model.get_phi_parameters():
        param.requires_grad = True # Phi params are trainable

    model.current_task_idx_internal = current_task_idx

    print(f"Training on Task {current_task_idx+1}...")
    for epoch in range(epochs):
        total_loss_epoch = 0
        correct_preds = 0
        total_target_points = 0
        num_batches_processed = 0

        for batch_idx, (x_batch_flat, y_batch_labels_flat) in enumerate(train_loader):
            x_batch_flat, y_batch_labels_flat = x_batch_flat.to(device), y_batch_labels_flat.to(device)
            current_b_size = x_batch_flat.size(0)
            x_for_split = x_batch_flat.view(1, current_b_size, -1)
            y_for_split = y_batch_labels_flat.view(1, current_b_size)

            x_context, y_context_onehot, _, x_target, y_target_labels, has_context = \
                get_context_target_split(x_for_split, y_for_split,
                                         num_context_range=(max(1, int(current_b_size * 0.1)), int(current_b_size * 0.7)),
                                         num_extra_target_range=(max(1, int(current_b_size * 0.2)), int(current_b_size * 0.9)),
                                         y_dim_onehot=model.y_dim_onehot)

            if not has_context or x_target.size(1) == 0:
                continue

            optimizer_phi.zero_grad(set_to_none=True) # More efficient
            optimizer_decoder.zero_grad(set_to_none=True)

            y_pred_logits, z_mean, z_logvar = model(x_context, y_context_onehot, x_target)
            loss, _, _ = np_loss(y_pred_logits, y_target_labels, z_mean, z_logvar, kl_weight=kl_weight)
            loss.backward()

            if model.past_task_jacobians_stacked is not None and model.past_task_jacobians_stacked.numel() > 0:
                 project_gradients_opm(model, reg=JACOBIAN_PROJ_REG, proj_dim=100) # proj_dim for J_old projection

            optimizer_phi.step()
            optimizer_decoder.step()

            total_loss_epoch += loss.item()
            _, predicted_labels = torch.max(y_pred_logits.data, -1)
            correct_preds += (predicted_labels == y_target_labels).sum().item()
            total_target_points += y_target_labels.numel()
            num_batches_processed += 1

        avg_loss = total_loss_epoch / num_batches_processed if num_batches_processed > 0 else 0
        accuracy = 100. * correct_preds / total_target_points if total_target_points > 0 else 0
        print(f'  Epoch {epoch+1}/{epochs}: Avg Loss: {avg_loss:.4f}, Acc: {accuracy:.2f}%')


def evaluate_np_task(model, task_id_to_eval, num_eval_batches=50, batch_size=64, fixed_num_context=20):
    x_test_task_np, y_test_task_np = task_data_test[task_id_to_eval]
    test_dataset = TensorDataset(x_test_task_np, y_test_task_np)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    model.eval()
    total_correct = 0
    total_targets = 0
    num_batches_processed = 0

    # print(f"Evaluating on Task {task_id_to_eval+1}...") # Moved outside loop for less verbose output
    with torch.no_grad():
        for i, (x_batch_flat, y_batch_labels_flat) in enumerate(test_loader):
            if i >= num_eval_batches: break
            x_batch_flat, y_batch_labels_flat = x_batch_flat.to(device), y_batch_labels_flat.to(device)
            current_b_size = x_batch_flat.size(0)
            x_for_split = x_batch_flat.view(1, current_b_size, -1)
            y_for_split = y_batch_labels_flat.view(1, current_b_size)

            num_context_eval = min(fixed_num_context, current_b_size -1)
            if num_context_eval < 0 : num_context_eval = 0
            if current_b_size <=1 : num_context_eval = 0
            num_target_eval = current_b_size - num_context_eval
            if num_target_eval <=0 : continue

            indices = torch.randperm(current_b_size, device=x_batch_flat.device)
            context_indices = indices[:num_context_eval]
            target_indices = indices[num_context_eval : num_context_eval + num_target_eval]
            x_context = x_for_split[:, context_indices, :]
            y_context_labels = y_for_split[:, context_indices]
            x_target = x_for_split[:, target_indices, :]
            y_target_labels = y_for_split[:, target_indices]
            y_context_onehot = F.one_hot(y_context_labels.long(), num_classes=model.y_dim_onehot).float()

            if num_context_eval == 0 and x_context.nelement() == 0: pass

            y_pred_logits, _, _ = model(x_context, y_context_onehot, x_target, task_id_for_decoder=task_id_to_eval)
            _, predicted = torch.max(y_pred_logits.data, -1)
            total_correct += (predicted == y_target_labels).sum().item()
            total_targets += y_target_labels.numel()
            num_batches_processed +=1
            
    accuracy = 100. * total_correct / total_targets if total_targets > 0 else 0
    # print(f"Finished evaluating Task {task_id_to_eval+1}. Acc: {accuracy:.2f}%")
    return accuracy

# --- Run Continual Learning Experiment ---
def run_continual_learning_np_opm_experiment():
    x_dim, y_dim_onehot, y_dim_out = 784, 10, 10
    r_dim, z_dim = 64, 32  # Smaller for faster Jacobian test
    enc_hidden_dim, dec_hidden_dim = 64, 64

    model = NeuralProcessOPM(x_dim, y_dim_onehot, r_dim, z_dim, enc_hidden_dim, dec_hidden_dim, y_dim_out).to(device)

    epochs_per_task = 5 # Reduced for speed
    batch_size_train = 32
    kl_weight_train = 0.01
    learning_rate = 1e-4

    task_accuracies = {i: [] for i in range(num_tasks)}
    avg_accuracies_over_time = []

    for current_task_idx in range(num_tasks):
        print(f"\n--- Task {current_task_idx+1}/{num_tasks} ---")
        if current_task_idx >= len(model.decoders):
            model.add_decoder_for_task()

        optimizer_phi = optim.Adam(model.get_phi_parameters(), lr=learning_rate)
        optimizer_decoder = optim.Adam(model.decoders[current_task_idx].parameters(), lr=learning_rate)

        train_np_task(model, current_task_idx, optimizer_phi, optimizer_decoder,
                      epochs=epochs_per_task, batch_size=batch_size_train, kl_weight=kl_weight_train)

        for param in model.decoders[current_task_idx].parameters():
            param.requires_grad = False # Freeze decoder for task k

        # Set phi_modules to eval mode for consistent Jacobian (e.g., if they had dropout/batchnorm)
        model.xy_encoder.eval()
        model.latent_encoder.eval()
        # Decoders are already either frozen or the current one is handled by task_id_of_decoder

        print(f"Collecting Jacobian for task {current_task_idx + 1}...")
        ctx_x_jac, ctx_y_jac, val_x_jac = get_opm_data_for_jacobian(
            current_task_idx, NUM_CONTEXT_JACOBIAN, M_JACOBIAN_SAMPLES, model.y_dim_onehot
        )

        if ctx_x_jac is not None and val_x_jac is not None and val_x_jac.size(1) > 0:
            J_i = collect_jacobian_for_task_opm_func(model, current_task_idx, ctx_x_jac, ctx_y_jac, val_x_jac)

            if J_i is not None and J_i.numel() > 0:
                print(f"  Jacobian J_{current_task_idx} collected, shape: {J_i.shape}")
                # Verify Jacobian's second dimension matches current total numel of phi_params
                current_phi_dim_flat = sum(p.numel() for p in model.get_phi_parameters())
                if J_i.shape[1] != current_phi_dim_flat:
                    print(f"CRITICAL ERROR during Jacobian collection: J_i.shape[1] ({J_i.shape[1]}) "
                          f"does not match current flat phi dimension ({current_phi_dim_flat}).")
                    # This indicates a mismatch in how phi_params_tuple was constructed in jacrev context
                    # vs. model.get_phi_parameters(). This needs to be identical.
                    # Skipping storing this Jacobian to prevent further errors.
                else:
                    if model.past_task_jacobians_stacked is None:
                        model.past_task_jacobians_stacked = J_i
                    else:
                        if model.past_task_jacobians_stacked.shape[1] != J_i.shape[1]:
                             raise ValueError(
                                 f"Phi dimension mismatch in Jacobians for concatenation! "
                                 f"Old: {model.past_task_jacobians_stacked.shape[1]}, New: {J_i.shape[1]}"
                             )
                        model.past_task_jacobians_stacked = torch.cat([model.past_task_jacobians_stacked, J_i], dim=0)
                    print(f"  Total past Jacobians stacked shape: {model.past_task_jacobians_stacked.shape if model.past_task_jacobians_stacked is not None else 'None'}")
            else:
                print(f"  Failed to collect Jacobian for task {current_task_idx + 1} (J_i is None or empty).")
        else:
            print(f"  Skipping Jacobian collection for task {current_task_idx + 1} due to insufficient data.")

        # Evaluation
        current_eval_accuracies = []
        print(f"\n--- Evaluating after Task {current_task_idx+1} training ---")
        for eval_task_idx in range(num_tasks): # Iterate through all tasks seen so far for evaluation
            if eval_task_idx <= current_task_idx:
                if eval_task_idx < len(model.decoders):
                    print(f"Evaluating on Task {eval_task_idx+1} (data from task {eval_task_idx})...")
                    acc = evaluate_np_task(model, eval_task_idx, num_eval_batches=30, batch_size=64, fixed_num_context=30)
                    task_accuracies[eval_task_idx].append(acc) # Accuracy of task 'eval_task_idx' after 'current_task_idx' is trained
                    current_eval_accuracies.append(acc)
                    print(f"  Accuracy on Task {eval_task_idx+1} (using its original data): {acc:.2f}%")
                else: # Should not happen
                    task_accuracies[eval_task_idx].append(np.nan)
                    current_eval_accuracies.append(np.nan)
            else: # Task not yet seen, append NaN for this stage for consistent list lengths
                task_accuracies[eval_task_idx].append(np.nan) # Or skip, but table plotting needs care

        if current_eval_accuracies:
             avg_acc = np.nanmean([acc for acc in current_eval_accuracies if not np.isnan(acc)]) # nanmean for robustness
             avg_accuracies_over_time.append(avg_acc)
             print(f"  Average accuracy over tasks seen so far: {avg_acc:.2f}%")

    return task_accuracies, avg_accuracies_over_time

# --- Run and Plot ---
print("Starting Orthogonal Projection Neural Process Continual Learning experiment (with torch.func)...")
task_accuracies_np_opm, avg_accuracies_np_opm = run_continual_learning_np_opm_experiment()

plt.figure(figsize=(12, 8)) # Adjusted size
for task_id in range(num_tasks):
    # task_accuracies[task_id] contains performance of task_id AFTER training task_id, task_id+1, ..., num_tasks-1
    # We need to pad with NaNs at the beginning for tasks trained BEFORE task_id
    
    # Number of stages task_id was evaluated on
    num_evaluations = len(task_accuracies_np_opm[task_id])
    
    # Create a list for plotting: [NaN, NaN, ..., acc_after_task_id, acc_after_task_id+1, ...]
    plot_values = [np.nan] * task_id + task_accuracies_np_opm[task_id]
    
    # Ensure the list is as long as num_tasks by padding with NaNs at the end if necessary
    # (e.g. if task_id is the last task, it's only evaluated once)
    if len(plot_values) < num_tasks:
        plot_values.extend([np.nan] * (num_tasks - len(plot_values)))
    
    plt.plot(range(1, num_tasks + 1), plot_values[:num_tasks], marker='o', linestyle='-', label=f'Task {task_id+1} Perf.')

if avg_accuracies_np_opm:
    plt.plot(range(1, len(avg_accuracies_np_opm) + 1), avg_accuracies_np_opm, marker='x', linestyle='--', color='k', linewidth=2, label='Avg. Accuracy') # K for black
plt.xlabel('Training Stage (After Training Task X)')
plt.ylabel('Accuracy (%)')
plt.title(f'OPM Neural Process (torch.func): Continual Learning ({num_tasks} Permuted MNIST Tasks)')
plt.xticks(range(1, num_tasks + 1))
plt.legend(loc='lower left', bbox_to_anchor=(0.0, 0.0)) # Adjusted legend
plt.grid(True, linestyle='--', alpha=0.7)
plt.ylim(0, 101)
plt.tight_layout() # Adjust layout
plt.savefig(f'np_opm_func_catastrophic_forgetting_{num_tasks}tasks.png')
plt.show()

if avg_accuracies_np_opm:
    print(f"\nFinal average accuracy across seen tasks (OPM with torch.func): {avg_accuracies_np_opm[-1]:.2f}%")
