import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import numpy as np
import matplotlib.pyplot as plt
import time
import argparse
from torch.func import functional_call, jacrev

# --- Configuration for Fixed Hyperparameters ---
class Config:
    # Model Dimensions (will be overridden by data_loader for x_dim, y_dims)
    X_DIM = 784
    Y_DIM_ONEHOT = 10
    Y_DIM_OUT = 10
    R_DIM = 64
    Z_DIM = 32
    ENC_HIDDEN_DIM = 64
    DEC_HIDDEN_DIM = 64

    # Training Hyperparameters
    LR = 1e-4
    KL_WEIGHT = 0.01
    BATCH_SIZE_TRAIN = 32
    BATCH_SIZE_EVAL = 64
    
    # Dynamic Head Hyperparameters
    Z_DIVERGENCE_THRESHOLD = 2.0 
    MAX_PROTOTYPES_PER_HEAD = 10
    MIN_SAMPLES_FOR_PROTOTYPE = 20
    NUM_Z_COLLECTIONS_FOR_PROTOTYPE = 5

    # OPM Hyperparameters
    M_JACOBIAN_SAMPLES = 100
    NUM_CONTEXT_JACOBIAN = 100
    JACOBIAN_PROJ_REG = 1e-5
    OPM_PROJ_DIM = 100

    # Evaluation Hyperparameters
    NUM_EVAL_BATCHES = 50
    FIXED_EVAL_CONTEXT = 30

    # Dataset Specific Fixed Params
    SPLIT_MNIST_CLASSES_PER_TASK = 2
    ROTATED_MNIST_MAX_ANGLE = 90.0
    DATA_ROOT = './data'

    # Set by CLI
    SEED = 42
    NUM_TASKS = 10 
    EPOCHS_PER_TASK = 5
    EXPERIMENT_TYPE = 'PermutedMNIST'
    FIXED_HEAD_PER_TASK = False
    DEVICE = torch.device("cpu") # Will be updated

# Instantiate global config
config = Config()

# Global Device and Seed Setup
device = torch.device("cpu") 
task_data_train_global = []
task_data_test_global = []

# --- Argument Parser ---
def parse_args_and_setup_config():
    global config, device
    parser = argparse.ArgumentParser(description="Orthogonal Gradient Projection Neural Process (OGP-NP) for Continual Learning.")
    parser.add_argument('--experiment_type', type=str, default=config.EXPERIMENT_TYPE,
                        choices=['PermutedMNIST', 'SplitMNIST', 'RotatedMNIST'],
                        help='Type of MNIST experiment.')
    parser.add_argument('--num_tasks', type=int, default=config.NUM_TASKS, 
                        help='Number of tasks.')
    parser.add_argument('--epochs_per_task', type=int, default=config.EPOCHS_PER_TASK, help='Epochs per task.')
    parser.add_argument('--seed', type=int, default=config.SEED, help='Random seed.')
    
    parser.add_argument('--fixed_head_per_task', action='store_true', default=config.FIXED_HEAD_PER_TASK,
                        help='If set, spawns a new head for each task. Overrides dynamic head allocation.')
    parser.add_argument('--z_divergence_threshold', type=float, default=config.Z_DIVERGENCE_THRESHOLD,
                        help='Symmetric KL Divergence threshold to spawn new head (if not fixed_head_per_task).')

    cli_args = parser.parse_args()

    config.SEED = cli_args.seed
    config.NUM_TASKS = cli_args.num_tasks
    config.EPOCHS_PER_TASK = cli_args.epochs_per_task
    config.EXPERIMENT_TYPE = cli_args.experiment_type
    config.FIXED_HEAD_PER_TASK = cli_args.fixed_head_per_task
    config.Z_DIVERGENCE_THRESHOLD = cli_args.z_divergence_threshold

    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.DEVICE = device 
    print(f"Global device set to: {device}")
    print(f"Running with configuration: {vars(config)}")

# --- Data Preparation ---
def prepare_task_data():
    global task_data_train_global, task_data_test_global, config
    task_data_train_global.clear(); task_data_test_global.clear()

    mnist_mean, mnist_std = (0.1307,), (0.3081,)
    normalize_transform = transforms.Normalize(mnist_mean, mnist_std)
    
    actual_num_cl_tasks = config.NUM_TASKS
    config.X_DIM = 28 * 28 

    if config.EXPERIMENT_TYPE == "PermutedMNIST":
        print("Loading Permuted MNIST...")
        config.Y_DIM_ONEHOT, config.Y_DIM_OUT = 10, 10
        full_transform = transforms.Compose([transforms.ToTensor(), normalize_transform])
        _mnist_train_full = torchvision.datasets.MNIST(root=config.DATA_ROOT, train=True, download=True, transform=full_transform)
        _mnist_test_full = torchvision.datasets.MNIST(root=config.DATA_ROOT, train=False, download=True, transform=full_transform)
        
        x_train_list = [_mnist_train_full[i][0].view(-1) for i in range(len(_mnist_train_full))]
        y_train_list = [_mnist_train_full[i][1] for i in range(len(_mnist_train_full))]
        x_train_base, y_train_base = torch.stack(x_train_list), torch.tensor(y_train_list)
        x_test_list = [_mnist_test_full[i][0].view(-1) for i in range(len(_mnist_test_full))]
        y_test_list = [_mnist_test_full[i][1] for i in range(len(_mnist_test_full))]
        x_test_base, y_test_base = torch.stack(x_test_list), torch.tensor(y_test_list)

        task_data_train_global.append((x_train_base, y_train_base))
        task_data_test_global.append((x_test_base, y_test_base))
        print(f"  PermutedMNIST Task 1 (Original): train {len(x_train_base)}, test {len(x_test_base)}")

        for i in range(1, actual_num_cl_tasks): 
            perm = np.random.permutation(config.X_DIM)
            task_data_train_global.append((x_train_base[:, perm], y_train_base))
            task_data_test_global.append((x_test_base[:, perm], y_test_base))
            print(f"  PermutedMNIST Task {i+1} (Permuted): train {len(x_train_base)}, test {len(x_test_base)}")

    elif config.EXPERIMENT_TYPE == "SplitMNIST":
        print(f"Loading Split MNIST (fixed {config.SPLIT_MNIST_CLASSES_PER_TASK} classes per task)...")
        config.Y_DIM_ONEHOT, config.Y_DIM_OUT = 10, 10
        if 10 % config.SPLIT_MNIST_CLASSES_PER_TASK != 0:
            raise ValueError("For SplitMNIST, 10 must be divisible by classes_per_split_fixed.")
        actual_num_cl_tasks = 10 // config.SPLIT_MNIST_CLASSES_PER_TASK
        print(f"  Derived number of CL tasks for SplitMNIST: {actual_num_cl_tasks}")

        mnist_train_raw = torchvision.datasets.MNIST(root=config.DATA_ROOT, train=True, download=True)
        mnist_test_raw = torchvision.datasets.MNIST(root=config.DATA_ROOT, train=False, download=True)
        x_train_01_chw = mnist_train_raw.data.float().unsqueeze(1) / 255.0 
        y_train_all = mnist_train_raw.targets
        x_test_01_chw = mnist_test_raw.data.float().unsqueeze(1) / 255.0
        y_test_all = mnist_test_raw.targets
        
        x_train_norm_chw = normalize_transform(x_train_01_chw)
        x_test_norm_chw = normalize_transform(x_test_01_chw)
        x_train_flat_all = x_train_norm_chw.view(-1, config.X_DIM)
        x_test_flat_all = x_test_norm_chw.view(-1, config.X_DIM)

        for i in range(actual_num_cl_tasks):
            task_labels = list(range(i * config.SPLIT_MNIST_CLASSES_PER_TASK, (i + 1) * config.SPLIT_MNIST_CLASSES_PER_TASK))
            train_mask = torch.isin(y_train_all, torch.tensor(task_labels))
            test_mask = torch.isin(y_test_all, torch.tensor(task_labels))
            task_data_train_global.append((x_train_flat_all[train_mask], y_train_all[train_mask]))
            task_data_test_global.append((x_test_flat_all[test_mask], y_test_all[test_mask]))
            print(f"  SplitMNIST Task {i+1}: labels {task_labels}, train {len(task_data_train_global[-1][0])}, test {len(task_data_test_global[-1][0])}")
    
    elif config.EXPERIMENT_TYPE == "RotatedMNIST":
        print(f"Loading Rotated MNIST (max angle {config.ROTATED_MNIST_MAX_ANGLE} for {actual_num_cl_tasks} tasks)...")
        config.Y_DIM_ONEHOT, config.Y_DIM_OUT = 10, 10
        
        mnist_train_raw = torchvision.datasets.MNIST(root=config.DATA_ROOT, train=True, download=True)
        mnist_test_raw = torchvision.datasets.MNIST(root=config.DATA_ROOT, train=False, download=True)
        x_train_01_chw = mnist_train_raw.data.float().unsqueeze(1) / 255.0
        y_train_all = mnist_train_raw.targets
        x_test_01_chw = mnist_test_raw.data.float().unsqueeze(1) / 255.0
        y_test_all = mnist_test_raw.targets
        
        current_max_angle = config.ROTATED_MNIST_MAX_ANGLE
        if actual_num_cl_tasks == 1:
            angles = [0.0] 
            if current_max_angle != 0.0:
                 print(f"  INFO: RotatedMNIST with 1 task, but max_angle is {current_max_angle}. Using 0 rotation for the single task.")
        elif actual_num_cl_tasks > 1:
            angles = [i * (current_max_angle / (actual_num_cl_tasks - 1)) for i in range(actual_num_cl_tasks)]
        else: angles = []

        for task_idx, angle in enumerate(angles):
            print(f"  RotatedMNIST Task {task_idx+1}: rotation {angle:.2f} deg.")
            rot_train_chw = TF.rotate(x_train_01_chw, angle, interpolation=TF.InterpolationMode.BILINEAR)
            rot_test_chw = TF.rotate(x_test_01_chw, angle, interpolation=TF.InterpolationMode.BILINEAR)
            norm_rot_train_chw = normalize_transform(rot_train_chw)
            norm_rot_test_chw = normalize_transform(rot_test_chw)
            task_data_train_global.append((norm_rot_train_chw.view(-1, config.X_DIM), y_train_all))
            task_data_test_global.append((norm_rot_test_chw.view(-1, config.X_DIM), y_test_all))
    else:
        raise ValueError(f"Unsupported experiment_type: {config.EXPERIMENT_TYPE}")
    
    config.NUM_CL_TASKS_EFFECTIVE = actual_num_cl_tasks

# --- Neural Process Model Components ---
class NPEncoder(nn.Module):
    def __init__(self, input_dim_x, input_dim_y, hidden_dim, r_dim):
        super(NPEncoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim_x + input_dim_y, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, r_dim))
    def forward(self, x, y): return self.fc(torch.cat([x, y], dim=-1))

class LatentEncoder(nn.Module):
    def __init__(self, r_dim, z_dim):
        super(LatentEncoder, self).__init__()
        self.fc_mean = nn.Linear(r_dim, z_dim)
        self.fc_logvar = nn.Linear(r_dim, z_dim)
    def forward(self, r_aggregated): return self.fc_mean(r_aggregated), self.fc_logvar(r_aggregated)

class NPDecoder(nn.Module):
    def __init__(self, x_dim, z_dim, hidden_dim, y_dim_out):
        super(NPDecoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(z_dim + x_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, y_dim_out))
    def forward(self, z_sample, x_target):
        num_target_points = x_target.size(1)
        z_repeated = z_sample.unsqueeze(1).repeat(1, num_target_points, 1)
        return self.fc(torch.cat([z_repeated, x_target], dim=-1))

# --- Divergence Helper Functions ---
def kl_divergence_diag_gaussians(mean1, logvar1, mean2, logvar2, eps=1e-8):
    var1 = torch.exp(logvar1) + eps
    var2 = torch.exp(logvar2) + eps
    log_var_ratio = logvar2 - logvar1
    kl_div = 0.5 * (torch.sum(var1 / var2, dim=-1) + \
                    torch.sum((mean2 - mean1).pow(2) / var2, dim=-1) - \
                    mean1.size(-1) + \
                    torch.sum(log_var_ratio, dim=-1))
    return kl_div

def symmetric_kl_divergence(mean1, logvar1, mean2, logvar2):
    kl_pq = kl_divergence_diag_gaussians(mean1, logvar1, mean2, logvar2)
    kl_qp = kl_divergence_diag_gaussians(mean2, logvar2, mean1, logvar1)
    return kl_pq + kl_qp

class OGPNP(nn.Module):
    def __init__(self, cfg_obj):
        super(OGPNP, self).__init__()
        self.cfg = cfg_obj
        self.x_dim, self.y_dim_onehot, self.y_dim_out = self.cfg.X_DIM, self.cfg.Y_DIM_ONEHOT, self.cfg.Y_DIM_OUT
        self.r_dim, self.z_dim = self.cfg.R_DIM, self.cfg.Z_DIM
        self.enc_hidden_dim, self.dec_hidden_dim = self.cfg.ENC_HIDDEN_DIM, self.cfg.DEC_HIDDEN_DIM

        self.xy_encoder = NPEncoder(self.x_dim, self.y_dim_onehot, self.enc_hidden_dim, self.r_dim)
        self.latent_encoder = LatentEncoder(self.r_dim, self.z_dim)
        self.phi_modules = nn.ModuleList([self.xy_encoder, self.latent_encoder])

        self.decoders = nn.ModuleList()
        self.decoder_optimizers = []
        self.task_prototypes_params = [] 
        self.head_task_counts = []
        self.task_to_head_map = {}
        
        self.past_task_jacobians_stacked = None
        self.device_internal = self.cfg.DEVICE 

    def to(self, *args, **kwargs): 
        super().to(*args, **kwargs)
        try:
            new_device = next(self.parameters()).device
            if self.device_internal != new_device:
                self.device_internal = new_device
                for i, proto_param_list in enumerate(self.task_prototypes_params):
                    self.task_prototypes_params[i] = [
                        (m.to(self.device_internal) if m.device != self.device_internal else m, 
                         lv.to(self.device_internal) if lv.device != self.device_internal else lv)
                        for m, lv in proto_param_list 
                        if isinstance(m, torch.Tensor) and isinstance(lv, torch.Tensor)
                    ]
        except StopIteration: 
            if args and isinstance(args[0], (str, torch.device)):
                 self.device_internal = torch.device(args[0])
            elif kwargs and 'device' in kwargs:
                 self.device_internal = torch.device(kwargs['device'])
        return self

    def get_phi_parameters(self):
        return [p for module in self.phi_modules for p in module.parameters()]

    def aggregate(self, r_i):
        return torch.mean(r_i, dim=1) if r_i.size(1) > 0 else torch.zeros(r_i.size(0), self.r_dim, device=r_i.device)

    def reparameterize(self, z_mean, z_logvar):
        std = torch.exp(0.5 * z_logvar); eps = torch.randn_like(std); return z_mean + eps * std

    @torch.no_grad()
    def get_task_z_distribution_params(self, x_context, y_context_onehot):
        original_modes = [self.xy_encoder.training, self.latent_encoder.training]
        self.xy_encoder.eval(); self.latent_encoder.eval()
        
        if x_context.numel() == 0 or x_context.size(1) == 0 :
            batch_size = x_context.size(0) if x_context.ndim > 1 and x_context.size(0) > 0 else 1
            z_mean = torch.zeros(batch_size, self.z_dim, device=self.device_internal)
            z_logvar = torch.zeros(batch_size, self.z_dim, device=self.device_internal)
        else:
            r_i = self.xy_encoder(x_context, y_context_onehot)
            r_agg = self.aggregate(r_i)
            z_mean, z_logvar = self.latent_encoder(r_agg)
        
        self.xy_encoder.train(original_modes[0]); self.latent_encoder.train(original_modes[1])
        return z_mean, z_logvar

    def _get_prototype_distribution(self, head_idx):
        if head_idx < len(self.task_prototypes_params) and self.task_prototypes_params[head_idx]:
            means = torch.stack([params[0] for params in self.task_prototypes_params[head_idx]])
            logvars = torch.stack([params[1] for params in self.task_prototypes_params[head_idx]])
            proto_mean = means.mean(dim=0)
            proto_logvar = logvars.mean(dim=0)
            return proto_mean, proto_logvar
        return None, None

    def _update_prototype_distribution(self, head_idx, new_z_mean, new_z_logvar):
        self.task_prototypes_params[head_idx].append(
            (new_z_mean.detach().clone().to(self.device_internal), 
             new_z_logvar.detach().clone().to(self.device_internal))
        )
        if len(self.task_prototypes_params[head_idx]) > self.cfg.MAX_PROTOTYPES_PER_HEAD:
            self.task_prototypes_params[head_idx].pop(0)

    def _spawn_new_head(self, initial_z_mean, initial_z_logvar):
        new_decoder = NPDecoder(self.x_dim, self.z_dim, self.dec_hidden_dim, self.y_dim_out).to(self.device_internal)
        self.decoders.append(new_decoder)
        new_head_idx = len(self.decoders) - 1
        
        optimizer_decoder = optim.Adam(new_decoder.parameters(), lr=self.cfg.LR)
        self.decoder_optimizers.append(optimizer_decoder)
        
        self.task_prototypes_params.append([])
        self._update_prototype_distribution(new_head_idx, initial_z_mean, initial_z_logvar)
        self.head_task_counts.append(1)
        print(f"Spawned new head {new_head_idx} with LR {self.cfg.LR}")
        return new_head_idx

    def decide_head_for_task(self, original_task_idx, task_context_x_batch, task_context_y_onehot_batch):
        avg_task_z_mean, avg_task_z_logvar = None, None
        if task_context_x_batch.size(1) >= self.cfg.MIN_SAMPLES_FOR_PROTOTYPE:
            current_task_z_means, current_task_z_logvars = [], []
            for _ in range(self.cfg.NUM_Z_COLLECTIONS_FOR_PROTOTYPE):
                z_m, z_lv = self.get_task_z_distribution_params(task_context_x_batch, task_context_y_onehot_batch)
                current_task_z_means.append(z_m.squeeze(0))
                current_task_z_logvars.append(z_lv.squeeze(0))
            avg_task_z_mean = torch.stack(current_task_z_means).mean(dim=0) if current_task_z_means else torch.zeros(self.z_dim, device=self.device_internal)
            avg_task_z_logvar = torch.stack(current_task_z_logvars).mean(dim=0) if current_task_z_logvars else torch.zeros(self.z_dim, device=self.device_internal)
        else:
            print(f"Task {original_task_idx+1}: Ctx size {task_context_x_batch.size(1)} < min_samples {self.cfg.MIN_SAMPLES_FOR_PROTOTYPE}. Using prior-like prototype for decision.")
            avg_task_z_mean = torch.zeros(self.z_dim, device=self.device_internal)
            avg_task_z_logvar = torch.zeros(self.z_dim, device=self.device_internal)

        if self.cfg.FIXED_HEAD_PER_TASK:
            new_head_idx = len(self.decoders) 
            print(f"Task {original_task_idx+1}: Fixed head allocation. Spawning new head {new_head_idx}.")
            assigned_head_idx = self._spawn_new_head(avg_task_z_mean, avg_task_z_logvar)
        else: 
            assigned_head_idx, min_divergence = -1, float('inf')
            if not self.decoders:
                print(f"Task {original_task_idx+1}: No existing decoders. Spawning head 0.")
                assigned_head_idx = self._spawn_new_head(avg_task_z_mean, avg_task_z_logvar)
            else:
                for h_idx_loop in range(len(self.decoders)):
                    proto_mean, proto_logvar = self._get_prototype_distribution(h_idx_loop)
                    if proto_mean is not None and proto_logvar is not None:
                        div = symmetric_kl_divergence(avg_task_z_mean, avg_task_z_logvar, proto_mean, proto_logvar)
                        if div < min_divergence:
                            min_divergence, assigned_head_idx = div, h_idx_loop
                
                if assigned_head_idx != -1 and min_divergence < self.cfg.Z_DIVERGENCE_THRESHOLD:
                    print(f"Task {original_task_idx+1}: Reusing head {assigned_head_idx}. Divergence: {min_divergence:.3f} (Thresh: {self.cfg.Z_DIVERGENCE_THRESHOLD})")
                    self._update_prototype_distribution(assigned_head_idx, avg_task_z_mean, avg_task_z_logvar)
                    self.head_task_counts[assigned_head_idx] += 1
                else:
                    reason = f"Min divergence {min_divergence:.3f} >= thresh" if assigned_head_idx != -1 else "No suitable head"
                    new_idx = self._spawn_new_head(avg_task_z_mean, avg_task_z_logvar)
                    print(f"Task {original_task_idx+1}: {reason}. Spawning new dynamic head {new_idx}.")
                    assigned_head_idx = new_idx
            
        self.task_to_head_map[original_task_idx] = assigned_head_idx
        return assigned_head_idx

    def forward(self, x_context, y_context_onehot, x_target, head_idx_for_forward):
        if not (0 <= head_idx_for_forward < len(self.decoders)):
             raise ValueError(f"Invalid head_idx {head_idx_for_forward} for forward. Decoders available: {len(self.decoders)}")
        r_i_ctx = self.xy_encoder(x_context, y_context_onehot)
        r_agg = self.aggregate(r_i_ctx)
        z_mean, z_logvar = self.latent_encoder(r_agg)
        z_sample = self.reparameterize(z_mean, z_logvar)
        return self.decoders[head_idx_for_forward](z_sample, x_target), z_mean, z_logvar

# --- Loss Function ---
def ogp_np_loss(y_pred_logits, y_target_labels, z_mean, z_logvar, kl_weight_cfg):
    y_target_labels = y_target_labels.long()
    ce = F.cross_entropy(y_pred_logits.view(-1, y_pred_logits.size(-1)), y_target_labels.view(-1), reduction='mean')
    kl = torch.mean(-0.5 * torch.sum(1 + z_logvar - z_mean.pow(2) - z_logvar.exp(), dim=1))
    return ce + kl_weight_cfg * kl, ce, kl

# --- Context/Target Split Helper ---
def get_context_target_split(x_sequence_batch, y_sequence_labels_batch, y_dim_onehot_cfg, cfg_obj):
    meta_batch_size, total_points, _ = x_sequence_batch.shape
    if meta_batch_size != 1:
        x_sequence_batch = x_sequence_batch[0:1]; y_sequence_labels_batch = y_sequence_labels_batch[0:1]

    if total_points == 0:
        return x_sequence_batch, torch.empty_like(x_sequence_batch.expand(-1,-1,y_dim_onehot_cfg)), y_sequence_labels_batch, \
               x_sequence_batch, y_sequence_labels_batch, False

    ctx_ratio_min, ctx_ratio_max = 0.1, 0.7 
    trg_ratio_min, trg_ratio_max = 0.2, 0.9 
    min_ctx_pts = max(1, int(total_points * ctx_ratio_min))
    max_ctx_pts = max(min_ctx_pts, int(total_points * ctx_ratio_max))
    if total_points == 1 : num_context = 0
    elif min_ctx_pts >= total_points : num_context = 0
    else: num_context = np.random.randint(min_ctx_pts, max_ctx_pts + 1) if min_ctx_pts <= max_ctx_pts else min_ctx_pts
    num_context = min(num_context, total_points -1 if total_points > 1 else 0); num_context = max(0, num_context)

    indices = torch.randperm(total_points, device=x_sequence_batch.device)
    context_indices = indices[:num_context]
    remaining_indices = indices[num_context:]
    num_remaining = remaining_indices.size(0)
    min_trg_pts = max(1, int(num_remaining * trg_ratio_min))
    max_trg_pts = max(min_trg_pts, int(num_remaining * trg_ratio_max))
    num_target = 0
    if num_remaining > 0:
        if min_trg_pts < max_trg_pts: num_target = np.random.randint(min_trg_pts, max_trg_pts + 1)
        elif min_trg_pts == max_trg_pts : num_target = min_trg_pts
        num_target = min(num_target, num_remaining)
    if num_target == 0 and num_remaining > 0: num_target = num_remaining
    if num_context == 0 and num_target == 0 and total_points > 0:
        num_target = total_points; target_indices = indices
        context_indices = torch.empty(0, dtype=torch.long, device=x_sequence_batch.device)
    else: target_indices = remaining_indices[:num_target]

    x_context = x_sequence_batch[:, context_indices, :]
    y_context_labels = y_sequence_labels_batch[:, context_indices]
    x_target = x_sequence_batch[:, target_indices, :]
    y_target_labels = y_sequence_labels_batch[:, target_indices]
    y_context_onehot = F.one_hot(y_context_labels.long(), num_classes=y_dim_onehot_cfg).float()
    
    has_valid_np_batch = (x_context.size(1) > 0 or x_target.size(1) > 0) 
    if x_target.size(1) == 0 and total_points > 0 : has_valid_np_batch = False
    return x_context, y_context_onehot, y_context_labels, x_target, y_target_labels, has_valid_np_batch

# --- OGP Specific Functions ---
@torch.no_grad()
def get_ogp_data_for_jacobian(task_id_for_data, cfg):
    # Ensure data is on CPU first, then move to cfg.DEVICE after indexing
    x_task_full_cpu, y_task_full_cpu = task_data_train_global[task_id_for_data]
    
    # Move to CPU if not already, for consistent indexing if indices are CPU
    x_task_full_cpu = x_task_full_cpu.cpu()
    y_task_full_cpu = y_task_full_cpu.cpu()

    total_samples = x_task_full_cpu.size(0)
    if total_samples == 0: return None, None, None

    num_ctx, num_val = cfg.NUM_CONTEXT_JACOBIAN, cfg.M_JACOBIAN_SAMPLES
    if num_ctx + num_val > total_samples:
        if total_samples <= num_val: num_val, num_ctx = total_samples, 0
        else: num_ctx = min(num_ctx, total_samples - num_val)
    if num_val == 0: return None, None, None
    
    # Create indices on CPU
    indices = torch.randperm(total_samples, device=torch.device("cpu")) 
    ctx_idxs = indices[:num_ctx]
    val_trg_idxs = indices[num_ctx : num_ctx + num_val]
    
    # Index CPU tensors with CPU indices, then move to target device
    ctx_x_jac = x_task_full_cpu[ctx_idxs].to(cfg.DEVICE).unsqueeze(0)
    ctx_y_lbls_jac = y_task_full_cpu[ctx_idxs].to(cfg.DEVICE).unsqueeze(0)
    ctx_y_onehot_jac = F.one_hot(ctx_y_lbls_jac.long(), num_classes=cfg.Y_DIM_ONEHOT).float().to(cfg.DEVICE) # Ensure one-hot is also on device
    
    trg_x_val_jac = x_task_full_cpu[val_trg_idxs].to(cfg.DEVICE).unsqueeze(0)

    if trg_x_val_jac.size(1) == 0: return None, None, None
    if num_ctx == 0:
        ctx_x_jac = torch.empty(1, 0, cfg.X_DIM, device=cfg.DEVICE, dtype=x_task_full_cpu.dtype)
        ctx_y_onehot_jac = torch.empty(1, 0, cfg.Y_DIM_ONEHOT, device=cfg.DEVICE, dtype=torch.float)
        
    return ctx_x_jac, ctx_y_onehot_jac, trg_x_val_jac

def collect_jacobian_for_task_ogp_func(model, original_task_idx_for_data,
                                       context_x_task_i, context_y_onehot_task_i, target_x_val_task_i, cfg):
    model_device = model.device_internal
    context_x_task_i = context_x_task_i.to(model_device)
    context_y_onehot_task_i = context_y_onehot_task_i.to(model_device)
    target_x_val_task_i = target_x_val_task_i.to(model_device)

    phi_param_tensors_ordered, phi_param_names_ordered = [], []
    xy_param_names_local, le_param_names_local = [], []

    for name, p in sorted(model.xy_encoder.named_parameters(), key=lambda item: item[0]):
        phi_param_tensors_ordered.append(p); phi_param_names_ordered.append(f"xy.{name}"); xy_param_names_local.append(name)
    num_params_xy_encoder = len(xy_param_names_local)
    buffers_xy_encoder = {name: buf.to(model_device) for name, buf in model.xy_encoder.named_buffers()}

    for name, p in sorted(model.latent_encoder.named_parameters(), key=lambda item: item[0]):
        phi_param_tensors_ordered.append(p); phi_param_names_ordered.append(f"le.{name}"); le_param_names_local.append(name)
    num_params_latent_encoder = len(le_param_names_local)
    buffers_latent_encoder = {name: buf.to(model_device) for name, buf in model.latent_encoder.named_buffers()}
    
    phi_params_tuple_for_jacrev = tuple(phi_param_tensors_ordered)

    head_idx_for_this_task = model.task_to_head_map.get(original_task_idx_for_data)
    if head_idx_for_this_task is None:
        print(f"Error OGP: Task {original_task_idx_for_data} no head for Jacobian. Skipping."); return torch.empty(0,0,device=model_device)
    
    decoder_module_for_jacobian = model.decoders[head_idx_for_this_task]

    def compute_outputs_for_encoders_phi_static(phi_params_rt_tuple, 
                                         ctx_x_s, ctx_y_onehot_s, trg_x_s,
                                         xy_enc_mod_s, le_enc_mod_s, agg_func_s, fixed_dec_mod_s,
                                         xy_p_names_s, le_p_names_s, xy_bufs_s, le_bufs_s, num_p_xy_s):
        
        params_xy_d = {name: phi_params_rt_tuple[idx] for idx, name in enumerate(xy_p_names_s)}
        params_le_d = {name: phi_params_rt_tuple[idx + num_p_xy_s] for idx, name in enumerate(le_p_names_s)}
        
        r_i = functional_call(xy_enc_mod_s, (params_xy_d, xy_bufs_s), args=(ctx_x_s, ctx_y_onehot_s))
        r_agg = agg_func_s(r_i)
        z_mean, _ = functional_call(le_enc_mod_s, (params_le_d, le_bufs_s), args=(r_agg,))
        y_pred_logits = fixed_dec_mod_s(z_mean, trg_x_s) 
        return y_pred_logits.flatten()

    model.xy_encoder.eval(); model.latent_encoder.eval(); decoder_module_for_jacobian.eval()
    J_i_tuple = jacrev(compute_outputs_for_encoders_phi_static, argnums=0, has_aux=False)(
        phi_params_tuple_for_jacrev, 
        context_x_task_i, context_y_onehot_task_i, target_x_val_task_i,
        model.xy_encoder, model.latent_encoder, model.aggregate, decoder_module_for_jacobian,
        xy_param_names_local, le_param_names_local, buffers_xy_encoder, buffers_latent_encoder, num_params_xy_encoder
    )

    J_i_flat_list = []
    if not J_i_tuple or len(J_i_tuple) != len(phi_params_tuple_for_jacrev):
        print(f"CRIT OGP Err: jacrev len mismatch. Exp {len(phi_params_tuple_for_jacrev)}, got {len(J_i_tuple) if J_i_tuple else 'None'}.")
        return torch.empty(0,0,device=model_device)

    total_out_dim = target_x_val_task_i.size(0)*target_x_val_task_i.size(1)*cfg.Y_DIM_OUT
    for i, J_p in enumerate(J_i_tuple):
        p_name = phi_param_names_ordered[i]
        if J_p is None:
            p_numel = phi_params_tuple_for_jacrev[i].numel()
            print(f"  Warn OGP: jacrev None for {p_name}. Zero fill ({total_out_dim},{p_numel}).")
            J_i_flat_list.append(torch.zeros(total_out_dim, p_numel, device=model_device))
            continue
        J_i_flat_list.append(J_p.reshape(J_p.shape[0], -1))
    
    if not J_i_flat_list: return torch.empty(0,0,device=model_device)
    return torch.cat(J_i_flat_list, dim=1).detach()

def project_gradients_ogp(model, cfg):
    if model.past_task_jacobians_stacked is None or model.past_task_jacobians_stacked.numel() == 0: return
    J_old, ogp_dev = model.past_task_jacobians_stacked, model.device_internal
    
    phi_params_w_grads = [p for p in model.get_phi_parameters() if p.grad is not None]
    if not phi_params_w_grads: return
    g_flat = torch.cat([p.grad.flatten() for p in phi_params_w_grads]).to(ogp_dev)

    if J_old.shape[1] != g_flat.shape[0]:
        print(f"CRIT OGP Dim Err: J_old.col ({J_old.shape[1]}) != grad.dim ({g_flat.shape[0]}). Skip proj."); return

    J_eff = J_old
    if J_old.size(0) > cfg.OPM_PROJ_DIM and cfg.OPM_PROJ_DIM > 0:
        n_constraints, proj_d = J_old.size(0), min(cfg.OPM_PROJ_DIM, J_old.size(0))
        rand_proj = torch.randn(n_constraints, proj_d, device=ogp_dev)
        q_rand, _ = torch.linalg.qr(rand_proj); J_eff = q_rand.T @ J_old

    A = J_eff @ J_eff.T; A.diagonal().add_(cfg.JACOBIAN_PROJ_REG); B = J_eff @ g_flat
    try:
        L = torch.linalg.cholesky(A); x = torch.cholesky_solve(B.unsqueeze(-1), L).squeeze(-1)
        g_proj_flat = g_flat - J_eff.T @ x
    except torch.linalg.LinAlgError:
        try: A_pinv = torch.linalg.pinv(A); x = A_pinv @ B; g_proj_flat = g_flat - J_eff.T @ x
        except torch.linalg.LinAlgError: g_proj_flat = g_flat; print("OGP: All solve fail. No proj.")
            
    offset = 0
    for p in phi_params_w_grads:
        numel = p.numel(); p.grad.data = g_proj_flat[offset:offset+numel].view_as(p.grad); offset+=numel

# --- Training and Evaluation Loops (using global config) ---
def train_ogp_np_task(model, original_task_idx, assigned_head_idx, optimizer_phi, cfg_obj):
    if not (0 <= assigned_head_idx < len(model.decoder_optimizers)):
        raise ValueError(f"Invalid head_idx {assigned_head_idx} or optimizers not setup for training.")
    optimizer_decoder_head = model.decoder_optimizers[assigned_head_idx]
    
    x_train_task, y_train_task = task_data_train_global[original_task_idx]
    if x_train_task.numel() == 0: print(f"Task {original_task_idx+1}: No training data."); return

    train_dataset = TensorDataset(x_train_task, y_train_task)
    bs_train = min(cfg_obj.BATCH_SIZE_TRAIN, len(train_dataset))
    if bs_train == 0: return
    train_loader = DataLoader(train_dataset, batch_size=bs_train, shuffle=True, drop_last=(len(train_dataset)>bs_train))
    
    model.train()
    for idx, dec_module in enumerate(model.decoders):
        dec_module.train(mode=(idx == assigned_head_idx))
        for param in dec_module.parameters(): param.requires_grad = (idx == assigned_head_idx)
    for param in model.get_phi_parameters(): param.requires_grad = True

    print(f"Training Original Task {original_task_idx+1} on Decoder Head {assigned_head_idx} (Epochs: {cfg_obj.EPOCHS_PER_TASK})...")
    for epoch in range(cfg_obj.EPOCHS_PER_TASK):
        ep_loss, ep_correct, ep_total_pts, batches_done = 0.0,0,0,0
        for x_dl_batch, y_dl_batch_labels in train_loader:
            x_dl_batch,y_dl_batch_labels = x_dl_batch.to(cfg_obj.DEVICE), y_dl_batch_labels.to(cfg_obj.DEVICE)
            if x_dl_batch.size(0)==0: continue
            n_pts_seq = x_dl_batch.size(0)
            x_seq_np,y_seq_lbl_np = x_dl_batch.reshape(1,n_pts_seq,-1), y_dl_batch_labels.reshape(1,n_pts_seq)
            x_ctx,y_ctx_onehot,_,x_trg,y_trg_lbls,valid_split = get_context_target_split(x_seq_np,y_seq_lbl_np,model.y_dim_onehot,cfg_obj)
            if not valid_split: continue

            optimizer_phi.zero_grad(set_to_none=True); optimizer_decoder_head.zero_grad(set_to_none=True)
            y_pred,z_m,z_lv = model(x_ctx,y_ctx_onehot,x_trg,head_idx_for_forward=assigned_head_idx)
            loss,_,_ = ogp_np_loss(y_pred,y_trg_lbls,z_m,z_lv,cfg_obj.KL_WEIGHT)
            if torch.isnan(loss) or torch.isinf(loss): print(f"Warn: NaN/Inf loss ep {epoch+1}. Skip."); continue
            
            loss.backward()
            if model.past_task_jacobians_stacked is not None and model.past_task_jacobians_stacked.numel() > 0:
                 project_gradients_ogp(model, cfg_obj)
            
            torch.nn.utils.clip_grad_norm_(model.get_phi_parameters(),1.0)
            torch.nn.utils.clip_grad_norm_(model.decoders[assigned_head_idx].parameters(),1.0)
            optimizer_phi.step(); optimizer_decoder_head.step()

            ep_loss+=loss.item(); _,preds=torch.max(y_pred.data,-1)
            ep_correct+=(preds.squeeze()==y_trg_lbls.squeeze()).sum().item()
            ep_total_pts+=y_trg_lbls.numel(); batches_done+=1
        
        if batches_done > 0: print(f"  Ep {epoch+1}: AvgLoss={ep_loss/batches_done:.4f}, Acc={100.*ep_correct/ep_total_pts:.2f}%")
        else: print(f"  Ep {epoch+1}: No valid batches.")

def evaluate_ogp_np_task(model, original_task_idx_to_eval, cfg_obj):
    x_test_task, y_test_task = task_data_test_global[original_task_idx_to_eval]
    if x_test_task.numel() == 0: return 0.0
    
    head_idx = model.task_to_head_map.get(original_task_idx_to_eval)
    if head_idx is None or not (0 <= head_idx < len(model.decoders)):
        print(f"Eval Warn: No/Invalid head for task {original_task_idx_to_eval+1}. Acc=0."); return 0.0
        
    test_dataset = TensorDataset(x_test_task, y_test_task)
    bs_eval = min(cfg_obj.BATCH_SIZE_EVAL, len(test_dataset))
    if bs_eval == 0: return 0.0
    test_loader = DataLoader(test_dataset,batch_size=bs_eval,shuffle=False,drop_last=(len(test_dataset)>bs_eval))

    model.eval()
    total_correct, total_pts = 0,0
    with torch.no_grad():
        for i, (x_dl_batch, y_dl_batch_labels) in enumerate(test_loader):
            if i >= cfg_obj.NUM_EVAL_BATCHES: break
            x_dl_batch,y_dl_batch_labels = x_dl_batch.to(cfg_obj.DEVICE), y_dl_batch_labels.to(cfg_obj.DEVICE)
            if x_dl_batch.size(0)==0: continue
            
            n_pts_seq = x_dl_batch.size(0)
            x_seq_np,y_seq_lbl_np = x_dl_batch.reshape(1,n_pts_seq,-1), y_dl_batch_labels.reshape(1,n_pts_seq)
            n_ctx = min(cfg_obj.FIXED_EVAL_CONTEXT, n_pts_seq-1 if n_pts_seq>1 else 0); n_ctx=max(0,n_ctx)
            n_trg = n_pts_seq - n_ctx
            if n_trg <= 0: continue

            ctx_idxs = torch.arange(n_ctx, device=cfg_obj.DEVICE)
            trg_idxs = torch.arange(n_ctx,n_pts_seq, device=cfg_obj.DEVICE) # Corrected target indices

            x_ctx,y_ctx_lbls = x_seq_np[:,ctx_idxs,:], y_seq_lbl_np[:,ctx_idxs]
            x_trg,y_trg_lbls = x_seq_np[:,trg_idxs,:], y_seq_lbl_np[:,trg_idxs]
            y_ctx_onehot = F.one_hot(y_ctx_lbls.long(),num_classes=model.y_dim_onehot).float()
            if n_ctx==0: x_ctx,y_ctx_onehot = torch.empty(1,0,model.x_dim,device=cfg_obj.DEVICE,dtype=x_seq_np.dtype), torch.empty(1,0,model.y_dim_onehot,device=cfg_obj.DEVICE,dtype=torch.float)
            if x_trg.size(1)==0: continue

            y_pred,_,_ = model(x_ctx,y_ctx_onehot,x_trg,head_idx_for_forward=head_idx)
            _,preds = torch.max(y_pred.data,-1)
            total_correct+=(preds.squeeze()==y_trg_lbls.squeeze()).sum().item(); total_pts+=y_trg_lbls.numel()
            
    return 100.*total_correct/total_pts if total_pts > 0 else 0.0

# --- Main Experiment Runner ---
def run_continual_learning_ogp_np_experiment(cfg_obj):
    model = OGPNP(cfg_obj).to(cfg_obj.DEVICE)
    optimizer_phi = optim.Adam(model.get_phi_parameters(), lr=cfg_obj.LR)

    task_accuracies = {i:[] for i in range(cfg_obj.NUM_CL_TASKS_EFFECTIVE)}
    avg_accs_stream, active_heads_stream = [],[]

    for original_task_id in range(cfg_obj.NUM_CL_TASKS_EFFECTIVE):
        print(f"\n--- Proc. Orig. Task {original_task_id+1}/{cfg_obj.NUM_CL_TASKS_EFFECTIVE} ({cfg_obj.EXPERIMENT_TYPE}) ---")
        
        model.eval() 
        x_train_curr, y_train_curr = task_data_train_global[original_task_id]
        n_dec_pts = min(x_train_curr.size(0), cfg_obj.NUM_CONTEXT_JACOBIAN, 100)
        
        assigned_head = -1
        if n_dec_pts >= cfg_obj.MIN_SAMPLES_FOR_PROTOTYPE and x_train_curr.size(0) > 0:
            idx_dec = torch.randperm(x_train_curr.size(0))[:n_dec_pts]
            ctx_x_d = x_train_curr[idx_dec].to(model.device_internal).unsqueeze(0)
            ctx_y_d_lbl = y_train_curr[idx_dec].to(model.device_internal).unsqueeze(0)
            ctx_y_d_onehot = F.one_hot(ctx_y_d_lbl.long(),num_classes=model.y_dim_onehot).float()
            assigned_head = model.decide_head_for_task(original_task_id, ctx_x_d, ctx_y_d_onehot)
        else:
            print(f"Task {original_task_id+1}: Data too small for prototype ({n_dec_pts} vs {cfg_obj.MIN_SAMPLES_FOR_PROTOTYPE}). Fallback head decision.")
            if cfg_obj.FIXED_HEAD_PER_TASK or not model.decoders:
                dummy_z_m = torch.zeros(model.z_dim,device=model.device_internal)
                dummy_z_lv = torch.zeros(model.z_dim,device=model.device_internal)
                assigned_head = model._spawn_new_head(dummy_z_m, dummy_z_lv)
            else: assigned_head = len(model.decoders)-1
            model.task_to_head_map[original_task_id] = assigned_head
            if assigned_head >= len(model.head_task_counts): model.head_task_counts.append(0) 
            model.head_task_counts[assigned_head] +=1


        active_heads_stream.append(len(model.decoders))
        print(f"Original Task {original_task_id+1} assigned to Head {assigned_head}.")

        train_ogp_np_task(model, original_task_id, assigned_head, optimizer_phi, cfg_obj)
        
        model.eval()
        print(f"Collecting J for orig. task {original_task_id+1} (head {assigned_head})...")
        ctx_x_j,ctx_y_j,val_x_j = get_ogp_data_for_jacobian(original_task_id,cfg_obj)
        if ctx_x_j is not None and val_x_j is not None and val_x_j.size(1)>0:
            J_i = collect_jacobian_for_task_ogp_func(model,original_task_id,ctx_x_j,ctx_y_j,val_x_j,cfg_obj)
            if J_i is not None and J_i.numel()>0:
                print(f"  J_task{original_task_id} (head {assigned_head}) shape: {J_i.shape}")
                phi_d = sum(p.numel() for p in model.get_phi_parameters())
                if J_i.shape[1]!=phi_d: print(f"  OGP J WARN: J dim {J_i.shape[1]}!=phi_dim {phi_d}")
                else:
                    if model.past_task_jacobians_stacked is None or model.past_task_jacobians_stacked.numel()==0:
                        model.past_task_jacobians_stacked = J_i
                    elif model.past_task_jacobians_stacked.shape[1]!=J_i.shape[1]:
                        print("OGP J WARN: Phi dim mismatch for J concat! Reset J_stacked."); model.past_task_jacobians_stacked=J_i
                    else: model.past_task_jacobians_stacked = torch.cat([model.past_task_jacobians_stacked,J_i],dim=0)
                if model.past_task_jacobians_stacked is not None: print(f"  Total J_stacked shape: {model.past_task_jacobians_stacked.shape}")
            else: print(f"  Failed to collect J for task {original_task_id+1}.")
        else: print(f"  Skipping J collection for task {original_task_id+1} (no OPM data).")

        current_eval_accs = []
        print(f"\n--- Evaluating after training Original Task {original_task_id+1} ---")
        for eval_id in range(cfg_obj.NUM_CL_TASKS_EFFECTIVE):
            if eval_id <= original_task_id:
                acc = evaluate_ogp_np_task(model, eval_id, cfg_obj)
                task_accuracies[eval_id].append(acc); current_eval_accs.append(acc)
                print(f"  Acc on Orig. Task {eval_id+1}: {acc:.2f}% (head {model.task_to_head_map.get(eval_id,'N/A')})")
            else: task_accuracies[eval_id].append(np.nan)
        
        if current_eval_accs:
            avg_acc = np.nanmean([a for a in current_eval_accs if not np.isnan(a)])
            avg_accs_stream.append(avg_acc)
            print(f"  Avg acc (tasks 1-{original_task_id+1}): {avg_acc:.2f}% | Active Heads: {len(model.decoders)}")

    return task_accuracies, avg_accs_stream, active_heads_stream

# --- Main Execution ---
if __name__ == '__main__':
    parse_args_and_setup_config() 
    prepare_task_data() 
    
    print(f"Starting OGP-NP CL: {config.EXPERIMENT_TYPE}, {config.NUM_CL_TASKS_EFFECTIVE} eff tasks. Fixed heads: {config.FIXED_HEAD_PER_TASK}")
    start_time = time.time()
    task_accs, avg_accs, heads_log = run_continual_learning_ogp_np_experiment(config)
    end_time = time.time()
    print(f"\nExperiment finished in {(end_time-start_time)/60:.2f} mins.")
    if heads_log: print(f"Final #heads: {heads_log[-1]}")

    # Plotting
    n_eff = config.NUM_CL_TASKS_EFFECTIVE
    fig,ax1 = plt.subplots(figsize=(13,8)); cmap=plt.cm.get_cmap('viridis',n_eff+2)
    for task_id in range(n_eff):
        accs = task_accs[task_id]; plot_vals=[np.nan]*task_id+accs
        if len(plot_vals)<n_eff: plot_vals.extend([np.nan]*(n_eff-len(plot_vals)))
        ax1.plot(range(1,n_eff+1),plot_vals[:n_eff],marker='o',ls='-',color=cmap(task_id),label=f'Task {task_id+1} Acc.')
    if avg_accs: ax1.plot(range(1,len(avg_accs)+1),avg_accs,marker='x',ls='--',color=cmap(n_eff),lw=2,label='Avg. Acc (Seen)')
    ax1.set_xlabel('Training Stage (After Orig. Task X)'); ax1.set_ylabel('Acc (%)',color='tab:blue')
    ax1.tick_params(axis='y',labelcolor='tab:blue'); ax1.set_xticks(range(1,n_eff+1)); ax1.set_ylim(-1,101); ax1.grid(True,ls=':',alpha=0.7)
    if heads_log:
        ax2=ax1.twinx(); color_h=cmap(n_eff+1)
        ax2.plot(range(1,len(heads_log)+1),heads_log,marker='s',ls=':',color=color_h,label='# Heads')
        ax2.set_ylabel('# Heads',color=color_h); ax2.tick_params(axis='y',labelcolor=color_h)
        min_h_plot = 0
        if heads_log: 
            min_h_val = min(heads_log) if heads_log else 1
            max_h_val = max(heads_log) if heads_log else 1
            min_h_plot = max(0, min_h_val -1)
            ax2.set_ylim(min_h_plot , max_h_val + 1) 
            ax2.set_yticks(np.arange(int(max(1,min_h_val)), int(max_h_val) + 2))


    fig.tight_layout(); lines1,labels1=ax1.get_legend_handles_labels()
    if heads_log: 
        lines2,labels2=ax2.get_legend_handles_labels()
        if labels2:
             ax1.legend(lines1+lines2, labels1+labels2, loc='best', fontsize='small')
        else:
             ax1.legend(lines1, labels1, loc='best', fontsize='small')
    else: ax1.legend(loc='best', fontsize='small')
    
    mode_str="FixedHead" if config.FIXED_HEAD_PER_TASK else "DynamicHeadSKL"
    title=f'OGP-NP ({config.EXPERIMENT_TYPE},{mode_str}): CL ({n_eff} Tasks,Seed {config.SEED})'
    plt.title(title); fname=f'ogp_np_{mode_str.lower()}_{config.EXPERIMENT_TYPE}_{n_eff}tasks_seed{config.SEED}.png'
    plt.savefig(fname); print(f"Plot saved: {fname}"); plt.show()

    print(f"\n--- OGP-NP Final Accuracies Table ({config.EXPERIMENT_TYPE}, {mode_str}) ---")
    hdr="Eval Task V|Train Stage->|"; [hdr:=hdr+f" {i+1:2d}    |" for i in range(n_eff)]; print(hdr); print("-" * len(hdr))
    for eval_id in range(n_eff):
        row=f"   Task {eval_id+1:2d}   | "; acc_list=task_accs[eval_id]
        for stage_idx in range(n_eff):
            if stage_idx<eval_id: row+="   -    | "
            else:
                acc_idx=stage_idx-eval_id
                if acc_idx<len(acc_list): row+=f"{acc_list[acc_idx]:7.2f} | " if not np.isnan(acc_list[acc_idx]) else "  N/A   | "
                else: row+="  ERR   | "
        print(row)
    if avg_accs: print(f"\nFinal avg acc: {avg_accs[-1]:.2f}%")

