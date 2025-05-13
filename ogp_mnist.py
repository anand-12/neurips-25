import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import numpy as np
import matplotlib.pyplot as plt # Keep for potential future use by user
import time
import argparse
from torch.func import functional_call, jacrev
import os
import json

# --- Configuration for Fixed Hyperparameters ---
class Config:
    # --- Data Dimensions ---
    X_DIM = 784  # Dimension of the input data (e.g., 28*28 for MNIST)
    Y_DIM_ONEHOT = 10  # Dimension of the one-hot encoded labels (e.g., 10 classes for MNIST)
    Y_DIM_OUT = 10  # Dimension of the model's output prediction (e.g., 10 classes)

    # --- Neural Process Architecture Dimensions ---
    R_DIM = 64  # Dimension of the deterministic representation r_i from NPEncoder
    Z_DIM = 32  # Dimension of the latent variable z from LatentEncoder
    ENC_HIDDEN_DIM = 64  # Hidden dimension for NPEncoder and LatentEncoder MLPs
    DEC_HIDDEN_DIM = 64  # Hidden dimension for NPDecoder MLP

    # --- Training Hyperparameters ---
    LR = 1e-4  # Learning rate for optimizers
    KL_WEIGHT = 0.01  # Weight for the KL divergence term in the NP loss
    BATCH_SIZE_TRAIN = 32  # Batch size for training
    BATCH_SIZE_EVAL = 64  # Batch size for evaluation
    EPOCHS_PER_TASK = 5  # Number of epochs to train on each task

    # --- Dynamic Head Allocation Hyperparameters ---
    THRESHOLD = 2.0  # Symmetric KL Divergence threshold to spawn a new head (if not fixed_head_per_task)
    MIN_SAMPLES_FOR_PROTOTYPE = 20  # Minimum context samples required to form a reliable Z-archetype
    NUM_Z_COLLECTIONS_FOR_PROTOTYPE = 5  # Number of Z samples to average for a task's archetype Z

    # --- Orthogonal Gradient Projection (OGP) Hyperparameters ---
    M_JACOBIAN_SAMPLES = 100  # Number of target samples (validation points) used for Jacobian calculation per task
    NUM_CONTEXT_JACOBIAN = 100  # Number of context points used for Jacobian calculation per task
    JACOBIAN_PROJ_REG = 1e-5  # Regularization added to (J @ J.T) for numerical stability in OGP
    OPM_PROJ_DIM = 100  # If J_old has more rows (constraints) than this, project J_old to this dimension for OGP

    # --- Evaluation Hyperparameters ---
    NUM_EVAL_BATCHES = 50  # Max number of batches to use during evaluation per task (for speed)
    FIXED_EVAL_CONTEXT = 30  # Number of context points to use from each batch during evaluation

    # --- Experiment Setup ---
    EXPERIMENT_TYPE = 'PermutedMNIST'  # Type of MNIST experiment: 'PermutedMNIST', 'SplitMNIST', 'RotatedMNIST'
    NUM_TASKS = 10  # Number of tasks for PermutedMNIST/RotatedMNIST (SplitMNIST derives its own)
    SPLIT_MNIST_CLASSES_PER_TASK = 2  # Number of classes per task for SplitMNIST
    ROTATED_MNIST_MAX_ANGLE = 90.0  # Maximum rotation angle for RotatedMNIST tasks
    
    DATA_ROOT = './data'  # Root directory for storing MNIST dataset
    SEED = 42  # Random seed for reproducibility
    FIXED_HEAD_PER_TASK = False  # If True, task N uses head N, overriding dynamic allocation
    DEVICE = torch.device("cpu") # Will be updated based on CUDA availability


config = Config() # Global config object
device = torch.device("cpu") # Global device, will be updated
task_data_train_global = [] # Global list to store training data for each task
task_data_test_global = []  # Global list to store test data for each task

def parse_args_and_setup_config():
    """Parses command-line arguments and updates the global config object and device."""
    global config, device
    parser = argparse.ArgumentParser(description="Orthogonal Gradient Projection Neural Process (OGP-NP) for Continual Learning.")
    parser.add_argument('--experiment_type', type=str, default=config.EXPERIMENT_TYPE,
                        choices=['PermutedMNIST', 'SplitMNIST', 'RotatedMNIST'],
                        help='Type of MNIST experiment.')
    parser.add_argument('--num_tasks', type=int, default=config.NUM_TASKS,
                        help='Number of tasks (for Permuted/Rotated). SplitMNIST derives its own based on classes_per_task.')
    parser.add_argument('--epochs_per_task', type=int, default=config.EPOCHS_PER_TASK, help='Epochs per task.')
    parser.add_argument('--seed', type=int, default=config.SEED, help='Random seed.')
    parser.add_argument('--fixed_head_per_task', action='store_true', default=config.FIXED_HEAD_PER_TASK,
                        help='If set, task N uses head N, overriding dynamic head allocation.')
    parser.add_argument('--threshold', type=float, default=config.THRESHOLD, 
                        help='Symmetric KL Divergence threshold to spawn new head (if not fixed_head_per_task).')
    cli_args = parser.parse_args()

    config.SEED = cli_args.seed
    config.NUM_TASKS = cli_args.num_tasks 
    config.EPOCHS_PER_TASK = cli_args.epochs_per_task
    config.EXPERIMENT_TYPE = cli_args.experiment_type
    config.FIXED_HEAD_PER_TASK = cli_args.fixed_head_per_task
    config.THRESHOLD = cli_args.threshold

    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.DEVICE = device
    print(f"Global device set to: {device}")
    print(f"Running with configuration: {vars(config)}")

def prepare_task_data():
    """Loads and preprocesses task data based on the experiment type."""
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
            raise ValueError("For SplitMNIST, 10 must be divisible by config.SPLIT_MNIST_CLASSES_PER_TASK.")
        
        actual_num_cl_tasks = 10 // config.SPLIT_MNIST_CLASSES_PER_TASK 
        print(f"  Derived number of CL tasks for SplitMNIST: {actual_num_cl_tasks}")
        
        mnist_train_raw = torchvision.datasets.MNIST(root=config.DATA_ROOT, train=True, download=True)
        mnist_test_raw = torchvision.datasets.MNIST(root=config.DATA_ROOT, train=False, download=True)
        
        x_train_01_chw = mnist_train_raw.data.float().unsqueeze(1)/255.0 
        y_train_all = mnist_train_raw.targets
        x_test_01_chw = mnist_test_raw.data.float().unsqueeze(1)/255.0
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
        
        x_train_01_chw = mnist_train_raw.data.float().unsqueeze(1)/255.0
        y_train_all = mnist_train_raw.targets
        x_test_01_chw = mnist_test_raw.data.float().unsqueeze(1)/255.0
        y_test_all = mnist_test_raw.targets
        
        current_max_angle = config.ROTATED_MNIST_MAX_ANGLE
        if actual_num_cl_tasks == 1:
            angles = [0.0] 
            if current_max_angle != 0.0: print(f"  INFO: RotatedMNIST with 1 task, using 0 rotation.")
        elif actual_num_cl_tasks > 1:
            angles = [i * (current_max_angle / (actual_num_cl_tasks - 1)) for i in range(actual_num_cl_tasks)]
        else: 
            angles = [] 
            
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
    """Encodes (x, y) pairs into deterministic representations r_i."""
    def __init__(self,input_dim, output_dim_onehot, hidden_dim, r_dim):
        super(NPEncoder,self).__init__()
        self.fc=nn.Sequential(
            nn.Linear(input_dim + output_dim_onehot, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, r_dim)
        )
    def forward(self,x,y_onehot): # x: (B, N_ctx, Dx), y_onehot: (B, N_ctx, Dy_onehot)
        return self.fc(torch.cat([x,y_onehot],dim=-1)) # Out: (B, N_ctx, Dr)

class LatentEncoder(nn.Module):
    """Encodes aggregated r_agg into latent variable z parameters (mean, logvar)."""
    def __init__(self,r_dim,z_dim):
        super(LatentEncoder,self).__init__()
        self.fc_mean=nn.Linear(r_dim,z_dim)
        self.fc_logvar=nn.Linear(r_dim,z_dim)
    def forward(self,r_aggregated): # r_aggregated: (B, Dr)
        return self.fc_mean(r_aggregated), self.fc_logvar(r_aggregated) # Out: (B, Dz), (B, Dz)

class NPDecoder(nn.Module):
    """Decodes latent z_sample and target x_target to predict y_target."""
    def __init__(self,x_dim,z_dim,hidden_dim,y_out_dim):
        super(NPDecoder,self).__init__()
        self.fc=nn.Sequential(
            nn.Linear(z_dim + x_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim,hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim,y_out_dim)
        )
    def forward(self,z_sample,x_target): # z_sample: (B, Dz), x_target: (B, N_trg, Dx)
        z_repeated = z_sample.unsqueeze(1).repeat(1,x_target.size(1),1) # (B, N_trg, Dz)
        return self.fc(torch.cat([z_repeated,x_target],dim=-1)) # Out: (B, N_trg, Dy_out)

# --- KL Divergence Utilities ---
def kl_divergence_diag_gaussians(mean1, logvar1, mean2, logvar2, epsilon=1e-8):
    """Computes KL(N(mean1, var1) || N(mean2, var2)) for diagonal Gaussians."""
    var1 = torch.exp(logvar1) + epsilon
    var2 = torch.exp(logvar2) + epsilon
    log_var_ratio = logvar2 - logvar1 # log(var2/var1)
    
    kl_div = 0.5 * (
        torch.sum(var1 / var2, dim=-1) + \
        torch.sum((mean2 - mean1).pow(2) / var2, dim=-1) - \
        mean1.size(-1) + \
        torch.sum(log_var_ratio, dim=-1)
    )
    return kl_div

def symmetric_kl_divergence(mean1, logvar1, mean2, logvar2):
    """Computes Symmetric KL = KL(P1||P2) + KL(P2||P1) for diagonal Gaussians."""
    kl_12 = kl_divergence_diag_gaussians(mean1, logvar1, mean2, logvar2)
    kl_21 = kl_divergence_diag_gaussians(mean2, logvar2, mean1, logvar1)
    return kl_12 + kl_21


# --- OGP-NP Model ---
class OGPNP(nn.Module):
    """Orthogonal Gradient Projection Neural Process model."""
    def __init__(self, cfg_obj):
        super(OGPNP, self).__init__()
        self.cfg = cfg_obj
        self.x_dim = cfg_obj.X_DIM
        self.y_dim_onehot = cfg_obj.Y_DIM_ONEHOT
        self.y_dim_out = cfg_obj.Y_DIM_OUT
        self.r_dim = cfg_obj.R_DIM
        self.z_dim = cfg_obj.Z_DIM
        self.enc_hidden_dim = cfg_obj.ENC_HIDDEN_DIM
        self.dec_hidden_dim = cfg_obj.DEC_HIDDEN_DIM
        
        self.xy_encoder = NPEncoder(self.x_dim, self.y_dim_onehot, self.enc_hidden_dim, self.r_dim)
        self.latent_encoder = LatentEncoder(self.r_dim, self.z_dim)
        self.phi_modules = nn.ModuleList([self.xy_encoder, self.latent_encoder])
        
        self.decoders = nn.ModuleList()
        self.decoder_optimizers = []
        self.head_archetype_z_params = [] 
        self.head_task_counts = [] 
        
        self.task_to_head_map = {} 
        self.past_task_jacobians_stacked = None
        self.device_internal = cfg_obj.DEVICE

    def to(self, *args, **kwargs):
        """Moves model to specified device and updates internal device record and archetypes."""
        super().to(*args, **kwargs)
        new_device_target = None
        if args and isinstance(args[0], (str, torch.device)): 
            new_device_target = torch.device(args[0])
        elif kwargs and 'device' in kwargs: 
            new_device_target = torch.device(kwargs['device'])

        if new_device_target is None: 
            try:
                new_device_target = next(self.parameters()).device
            except StopIteration: 
                pass 

        if new_device_target and self.device_internal != new_device_target:
            self.device_internal = new_device_target
            self.head_archetype_z_params = [
                (m.to(self.device_internal), lv.to(self.device_internal))
                for m, lv in self.head_archetype_z_params
                if isinstance(m, torch.Tensor) and isinstance(lv, torch.Tensor)
            ]
        return self

    def get_phi_parameters(self):
        """Returns a list of all parameters in the shared phi_modules (encoders)."""
        return [p for module in self.phi_modules for p in module.parameters()]

    def aggregate(self, r_i): 
        """Aggregates deterministic representations r_i (typically by mean)."""
        return torch.mean(r_i, dim=1) if r_i.size(1) > 0 else torch.zeros(r_i.size(0), self.r_dim, device=r_i.device)

    def reparameterize(self, z_mean, z_logvar): 
        """Reparameterization trick to sample from N(z_mean, exp(z_logvar))."""
        std_dev = torch.exp(0.5 * z_logvar)
        epsilon = torch.randn_like(std_dev) 
        return z_mean + epsilon * std_dev 

    @torch.no_grad()
    def get_task_z_distribution_params(self, x_ctx, y_ctx_onehot):
        """Computes parameters (mean, logvar) of the latent Z for given context data."""
        original_modes = [self.xy_encoder.training, self.latent_encoder.training]
        self.xy_encoder.eval(); self.latent_encoder.eval() 

        if x_ctx.numel() == 0 or x_ctx.size(1) == 0: 
            batch_size = x_ctx.size(0) if x_ctx.ndim > 1 and x_ctx.size(0) > 0 else 1
            z_mean = torch.zeros(batch_size, self.z_dim, device=self.device_internal)
            z_logvar = torch.zeros(batch_size, self.z_dim, device=self.device_internal) 
        else:
            r_i = self.xy_encoder(x_ctx, y_ctx_onehot)
            r_agg = self.aggregate(r_i)
            z_mean, z_logvar = self.latent_encoder(r_agg)
        
        self.xy_encoder.train(original_modes[0]); self.latent_encoder.train(original_modes[1])
        return z_mean, z_logvar 

    def _spawn_new_head(self, archetypal_zm_for_new_head, archetypal_zlv_for_new_head):
        """Creates and initializes a new decoder head and its optimizer."""
        new_decoder = NPDecoder(self.x_dim, self.z_dim, self.dec_hidden_dim, self.y_dim_out).to(self.device_internal)
        self.decoders.append(new_decoder)
        new_head_idx = len(self.decoders) - 1

        optimizer_decoder = optim.Adam(new_decoder.parameters(), lr=self.cfg.LR)
        self.decoder_optimizers.append(optimizer_decoder)

        self.head_archetype_z_params.append(
            (archetypal_zm_for_new_head.detach().clone(), 
             archetypal_zlv_for_new_head.detach().clone())
        )
        self.head_task_counts.append(1) 
        return new_head_idx

    def decide_head_for_task(self, orig_task_idx, task_ctx_x, task_ctx_y_onehot):
        """Decides which head to use for a task, spawning a new one if necessary.
           Returns: (assigned_head_idx, action_string)
        """
        assigned_head_idx = -1
        action_string = "unknown"

        if self.cfg.FIXED_HEAD_PER_TASK:
            assigned_head_idx = orig_task_idx 
            action_string = "fixed_assignment_ensured_exists"
            while len(self.decoders) <= assigned_head_idx:
                dummy_zm = torch.zeros(self.z_dim, device=self.device_internal)
                dummy_zlv = torch.zeros(self.z_dim, device=self.device_internal)
                self._spawn_new_head(dummy_zm, dummy_zlv) # This also appends to head_archetype_z_params and head_task_counts
            print(f"Task {orig_task_idx+1}: Fixed head assignment to Head {assigned_head_idx}.")
        else: 
            current_task_azm_single, current_task_azlv_single = None, None
            if task_ctx_x.size(1) >= self.cfg.MIN_SAMPLES_FOR_PROTOTYPE:
                collected_z_means, collected_z_logvars = [], []
                for _ in range(self.cfg.NUM_Z_COLLECTIONS_FOR_PROTOTYPE):
                    zm_batch, zlv_batch = self.get_task_z_distribution_params(task_ctx_x, task_ctx_y_onehot)
                    collected_z_means.append(zm_batch.squeeze(0)) 
                    collected_z_logvars.append(zlv_batch.squeeze(0))

                if collected_z_means: 
                    current_task_azm_single = torch.stack(collected_z_means).mean(dim=0)
                    current_task_azlv_single = torch.stack(collected_z_logvars).mean(dim=0)
                else: 
                    current_task_azm_single = torch.zeros(self.z_dim, device=self.device_internal)
                    current_task_azlv_single = torch.zeros(self.z_dim, device=self.device_internal)
            else: 
                print(f"Task {orig_task_idx+1}: Context size {task_ctx_x.size(1)} < min_samples {self.cfg.MIN_SAMPLES_FOR_PROTOTYPE}. Using prior-like Z for head decision.")
                current_task_azm_single = torch.zeros(self.z_dim, device=self.device_internal)
                current_task_azlv_single = torch.zeros(self.z_dim, device=self.device_internal)
            
            current_task_azm_single = current_task_azm_single.detach().clone().to(self.device_internal)
            current_task_azlv_single = current_task_azlv_single.detach().clone().to(self.device_internal)

            if not self.decoders: 
                assigned_head_idx = self._spawn_new_head(current_task_azm_single, current_task_azlv_single)
                action_string = "dynamic_spawned_first_head"
                print(f"Task {orig_task_idx+1}: No heads exist (dynamic). Spawning head {assigned_head_idx} with current task's Z as archetype.")
            else: 
                best_matching_head_idx = -1
                min_kl_divergence = float('inf')
                for head_idx_iter in range(len(self.decoders)):
                    archetype_zm, archetype_zlv = self.head_archetype_z_params[head_idx_iter]
                    kl_div_tensor = symmetric_kl_divergence(
                        current_task_azm_single.unsqueeze(0), current_task_azlv_single.unsqueeze(0),
                        archetype_zm.unsqueeze(0), archetype_zlv.unsqueeze(0)
                    )
                    kl_div = kl_div_tensor.item() 
                    if kl_div < min_kl_divergence:
                        min_kl_divergence = kl_div
                        best_matching_head_idx = head_idx_iter
                
                if best_matching_head_idx != -1 and min_kl_divergence < self.cfg.THRESHOLD:
                    assigned_head_idx = best_matching_head_idx
                    self.head_task_counts[assigned_head_idx] += 1 
                    action_string = "dynamic_reused_archetype"
                    print(f"Task {orig_task_idx+1}: Reusing head {assigned_head_idx} (matches archetype Z). Divergence: {min_kl_divergence:.3f} (Threshold: {self.cfg.THRESHOLD})")
                else: 
                    reason = f"Min divergence {min_kl_divergence:.3f} >= threshold" if best_matching_head_idx != -1 else "No suitable archetype found"
                    assigned_head_idx = self._spawn_new_head(current_task_azm_single, current_task_azlv_single)
                    action_string = "dynamic_spawned_new_archetype"
                    print(f"Task {orig_task_idx+1}: {reason}. Spawning new head {assigned_head_idx} with current task's Z as archetype.")
        
        self.task_to_head_map[orig_task_idx] = assigned_head_idx
        return assigned_head_idx, action_string

    def forward(self, x_context, y_context_onehot, x_target, head_idx):
        """Forward pass through the NP using a specific decoder head."""
        if not (0 <= head_idx < len(self.decoders)):
             raise ValueError(f"Invalid head_idx {head_idx} for forward. Decoders available: {len(self.decoders)}, requested: {head_idx}")
        r_i_ctx = self.xy_encoder(x_context, y_context_onehot)
        r_agg = self.aggregate(r_i_ctx)
        z_mean, z_logvar = self.latent_encoder(r_agg)
        z_sample = self.reparameterize(z_mean, z_logvar)
        return self.decoders[head_idx](z_sample, x_target), z_mean, z_logvar


# --- Loss Function ---
def ogp_np_loss(y_pred,y_trg_lbl,zm,zlv,kl_w):
    """Computes the OGP-NP loss (Cross-Entropy + KL Divergence)."""
    y_trg_lbl=y_trg_lbl.long() 
    ce_loss = F.cross_entropy(y_pred.reshape(-1,y_pred.size(-1)),y_trg_lbl.reshape(-1),reduction='mean')
    kl_individual_samples = -0.5 * torch.sum(1 + zlv - zm.pow(2) - zlv.exp(), dim=1) 
    kl_term = torch.mean(kl_individual_samples) 
    
    total_loss = ce_loss + kl_w * kl_term
    return total_loss, ce_loss, kl_term

# --- Data Utilities ---
def get_context_target_split(x_seq_batch, y_seq_labels_batch, y_dim_onehot_cfg, cfg_obj):
    """Splits a batch of sequences into context and target sets for NP training."""
    _, total_points_in_sequence, _ = x_seq_batch.shape
    if total_points_in_sequence == 0: 
        return (torch.empty_like(x_seq_batch.expand(-1, 0, -1)), 
                torch.empty_like(x_seq_batch.expand(-1, 0, y_dim_onehot_cfg)), 
                torch.empty_like(y_seq_labels_batch.expand(-1, 0)), 
                torch.empty_like(x_seq_batch.expand(-1, 0, -1)), 
                torch.empty_like(y_seq_labels_batch.expand(-1, 0)), 
                False) 

    ctx_ratio_min, ctx_ratio_max = 0.1, 0.7 
    min_ctx_pts = max(1, int(total_points_in_sequence * ctx_ratio_min))
    max_ctx_pts = max(min_ctx_pts, int(total_points_in_sequence * ctx_ratio_max))
    max_ctx_pts = min(max_ctx_pts, total_points_in_sequence - 1 if total_points_in_sequence > 1 else 0)

    num_context = 0
    if total_points_in_sequence == 1: num_context = 0 
    elif min_ctx_pts > max_ctx_pts : num_context = max_ctx_pts 
    else: num_context = np.random.randint(min_ctx_pts, max_ctx_pts + 1)
    num_context = max(0, min(num_context, total_points_in_sequence -1 if total_points_in_sequence > 1 else 0))
    
    indices = torch.randperm(total_points_in_sequence, device=x_seq_batch.device)
    ctx_indices, target_indices = indices[:num_context], indices[num_context:]
    
    if target_indices.numel() == 0 and total_points_in_sequence > 0: 
        if num_context > 0 : 
            target_indices = ctx_indices[-1:].clone() 
            ctx_indices = ctx_indices[:-1] 
            num_context -=1
        else: 
            return get_context_target_split(x_seq_batch,y_seq_labels_batch,y_dim_onehot_cfg,cfg_obj)

    x_ctx, y_ctx_lbl = x_seq_batch[:,ctx_indices,:], y_seq_labels_batch[:,ctx_indices]
    x_trg, y_trg_lbl = x_seq_batch[:,target_indices,:], y_seq_labels_batch[:,target_indices]
    y_ctx_onehot = F.one_hot(y_ctx_lbl.long(), num_classes=y_dim_onehot_cfg).float() if num_context > 0 else \
                   torch.empty(1,0,y_dim_onehot_cfg, device=x_seq_batch.device, dtype=torch.float)
    
    valid_split = (x_trg.size(1) > 0) 
    return x_ctx,y_ctx_onehot,y_ctx_lbl,x_trg,y_trg_lbl,valid_split

@torch.no_grad()
def get_ogp_data_for_jacobian(task_id,cfg_obj):
    """Prepares context and target data for Jacobian calculation for a given task."""
    x_cpu,y_cpu=task_data_train_global[task_id];x_cpu,y_cpu=x_cpu.cpu(),y_cpu.cpu() 
    total_samples=x_cpu.size(0)
    if total_samples==0: return None,None,None
    
    num_ctx_jac, num_val_jac =cfg_obj.NUM_CONTEXT_JACOBIAN,cfg_obj.M_JACOBIAN_SAMPLES
    if num_ctx_jac + num_val_jac > total_samples:
        if total_samples <= num_val_jac : 
            num_val_jac, num_ctx_jac = total_samples, 0
        else: 
            num_ctx_jac = min(num_ctx_jac, total_samples - num_val_jac)
    if num_val_jac == 0: return None,None,None 

    indices_cpu = torch.randperm(total_samples, device=torch.device("cpu"))
    ctx_indices, val_indices = indices_cpu[:num_ctx_jac], indices_cpu[num_ctx_jac : num_ctx_jac + num_val_jac]
    
    ctx_x = x_cpu[ctx_indices].to(cfg_obj.DEVICE).unsqueeze(0) 
    ctx_y_lbl = y_cpu[ctx_indices].to(cfg_obj.DEVICE).unsqueeze(0) 
    ctx_y_onehot = F.one_hot(ctx_y_lbl.long(), num_classes=cfg_obj.Y_DIM_ONEHOT).float().to(cfg_obj.DEVICE)
    trg_x = x_cpu[val_indices].to(cfg_obj.DEVICE).unsqueeze(0) 

    if num_ctx_jac == 0:
        ctx_x = torch.empty(1, 0, cfg_obj.X_DIM, device=cfg_obj.DEVICE, dtype=x_cpu.dtype)
        ctx_y_onehot = torch.empty(1, 0, cfg_obj.Y_DIM_ONEHOT, device=cfg_obj.DEVICE, dtype=torch.float)
    return ctx_x,ctx_y_onehot,trg_x


# --- OGP Jacobian Calculation and Projection ---
def collect_jacobian_for_task_ogp_func(model,orig_task_id,ctx_x_i,ctx_y_oh_i,trg_x_val_i,cfg_obj):
    """Collects the Jacobian of decoder output w.r.t. shared encoder (phi) parameters."""
    model_device = model.device_internal
    ctx_x_i = ctx_x_i.to(model_device); ctx_y_oh_i = ctx_y_oh_i.to(model_device); trg_x_val_i = trg_x_val_i.to(model_device)

    phi_param_tensors, phi_param_names_global = [], []
    xy_encoder_param_names_local, latent_encoder_param_names_local = [], []

    for name, param in sorted(model.xy_encoder.named_parameters(), key=lambda item: item[0]):
        phi_param_tensors.append(param); phi_param_names_global.append(f"xy_encoder.{name}"); xy_encoder_param_names_local.append(name)
    num_phi_xy_encoder = len(xy_encoder_param_names_local)
    buffers_xy_encoder = {name: buffer.to(model_device) for name, buffer in model.xy_encoder.named_buffers()}

    for name, param in sorted(model.latent_encoder.named_parameters(), key=lambda item: item[0]):
        phi_param_tensors.append(param); phi_param_names_global.append(f"latent_encoder.{name}"); latent_encoder_param_names_local.append(name)
    buffers_latent_encoder = {name: buffer.to(model_device) for name, buffer in model.latent_encoder.named_buffers()}

    phi_params_tuple_for_jac = tuple(phi_param_tensors) 
    head_idx_for_task = model.task_to_head_map.get(orig_task_id)

    if head_idx_for_task is None or not (0 <= head_idx_for_task < len(model.decoders)):
        print(f"Error OGP: Task {orig_task_id+1} no valid head ({head_idx_for_task}). Skip J."); return torch.empty(0,0,device=model_device) 
    
    decoder_module_for_jac = model.decoders[head_idx_for_task] 

    def compute_output_from_phi_params_static(
            phi_params_runtime_tuple, context_x, context_y_onehot, target_x_val,
            xy_enc_module, latent_enc_module, aggregate_func,
            fixed_decoder_module, 
            xy_param_names_order, latent_param_names_order,
            xy_buffers_dict, latent_buffers_dict, num_xy_params_len):
        
        params_xy_encoder_dict = {name: phi_params_runtime_tuple[i] for i, name in enumerate(xy_param_names_order)}
        params_latent_encoder_dict = {name: phi_params_runtime_tuple[i+num_xy_params_len] for i, name in enumerate(latent_param_names_order)}

        r_i = functional_call(xy_enc_module, (params_xy_encoder_dict, xy_buffers_dict), args=(context_x, context_y_onehot))
        r_agg = aggregate_func(r_i)
        z_mean, _ = functional_call(latent_enc_module, (params_latent_encoder_dict, latent_buffers_dict), args=(r_agg))
        
        y_pred_flat = fixed_decoder_module(z_mean, target_x_val).flatten() 
        return y_pred_flat

    model.xy_encoder.eval(); model.latent_encoder.eval(); decoder_module_for_jac.eval()

    jacobian_tuple_per_phi_param = jacrev(compute_output_from_phi_params_static, argnums=0, has_aux=False)(
                                        phi_params_tuple_for_jac,
                                        ctx_x_i, ctx_y_oh_i, trg_x_val_i,
                                        model.xy_encoder, model.latent_encoder, model.aggregate,
                                        decoder_module_for_jac, 
                                        xy_encoder_param_names_local, latent_encoder_param_names_local,
                                        buffers_xy_encoder, buffers_latent_encoder, num_phi_xy_encoder)

    jacobian_matrices_flattened_list = []
    if not jacobian_tuple_per_phi_param or len(jacobian_tuple_per_phi_param) != len(phi_params_tuple_for_jac):
        print(f"CRITICAL OGP Error: jacrev returned unexpected number of elements. Expected {len(phi_params_tuple_for_jac)}, Got {len(jacobian_tuple_per_phi_param) if jacobian_tuple_per_phi_param else 'None'}.")
        return torch.empty(0,0,device=model_device)

    total_output_dim = trg_x_val_i.size(0) * trg_x_val_i.size(1) * cfg_obj.Y_DIM_OUT 

    for i, jac_for_one_param_tensor in enumerate(jacobian_tuple_per_phi_param):
        param_name_full = phi_param_names_global[i]
        if jac_for_one_param_tensor is None: 
            original_param_numel = phi_params_tuple_for_jac[i].numel()
            print(f"  Warning OGP: jacrev returned None for phi_param '{param_name_full}'. Filling with zeros ({total_output_dim}, {original_param_numel}).")
            jacobian_matrices_flattened_list.append(torch.zeros(total_output_dim, original_param_numel, device=model_device))
        else:
            jacobian_matrices_flattened_list.append(jac_for_one_param_tensor.reshape(jac_for_one_param_tensor.shape[0], -1))
    
    if not jacobian_matrices_flattened_list: return torch.empty(0,0,device=model_device)
    
    full_jacobian_matrix_task_i = torch.cat(jacobian_matrices_flattened_list, dim=1).detach()
    return full_jacobian_matrix_task_i


def project_gradients_ogp(model,cfg_obj):
    """Projects gradients of phi_modules onto the null space of past task Jacobians."""
    if model.past_task_jacobians_stacked is None or model.past_task_jacobians_stacked.numel()==0:return
    
    J_old = model.past_task_jacobians_stacked 
    ogp_device = model.device_internal
    phi_params_with_grads = [p for p in model.get_phi_parameters() if p.grad is not None]
    if not phi_params_with_grads: return 

    current_gradients_flat = torch.cat([p.grad.flatten() for p in phi_params_with_grads]).to(ogp_device)

    if J_old.shape[1] != current_gradients_flat.shape[0]:
        print(f"CRITICAL OGP Dim Error: J_old.col ({J_old.shape[1]}) != grad.dim ({current_gradients_flat.shape[0]}). Skipping projection.")
        return

    J_effective = J_old
    if J_old.size(0) > cfg_obj.OPM_PROJ_DIM and cfg_obj.OPM_PROJ_DIM > 0:
        num_constraints = J_old.size(0)
        projection_dim = min(cfg_obj.OPM_PROJ_DIM, J_old.size(0), J_old.size(1)) 
        if projection_dim > 0 :
            try:
                random_projection_matrix = torch.randn(num_constraints, projection_dim, device=ogp_device)
                q_matrix, _ = torch.linalg.qr(random_projection_matrix) 
                J_effective = q_matrix.T @ J_old 
            except torch.linalg.LinAlgError as e: 
                J_effective = J_old; print(f"  OGP: QR decomposition failed for random projection: {e}. Using full J_old.")
        else: J_effective = J_old 
            
    A_matrix = J_effective @ J_effective.T 
    A_matrix.diagonal().add_(cfg_obj.JACOBIAN_PROJ_REG) 
    B_vector = J_effective @ current_gradients_flat 

    projected_gradients_flat = current_gradients_flat 
    try: 
        L_cholesky = torch.linalg.cholesky(A_matrix)
        x_solution = torch.cholesky_solve(B_vector.unsqueeze(-1), L_cholesky).squeeze(-1)
        projected_gradients_flat = current_gradients_flat - J_effective.T @ x_solution
    except torch.linalg.LinAlgError:
        print("  OGP: Cholesky solve failed. Trying pseudo-inverse.")
        try: 
            A_pseudo_inv = torch.linalg.pinv(A_matrix)
            x_solution = A_pseudo_inv @ B_vector
            projected_gradients_flat = current_gradients_flat - J_effective.T @ x_solution
        except torch.linalg.LinAlgError: 
            print("  OGP: All linear solves failed for gradient projection. No projection applied.")
            
    offset = 0
    for p in phi_params_with_grads:
        numel = p.numel()
        p.grad.data = projected_gradients_flat[offset:offset+numel].view_as(p.grad.data)
        offset += numel

# --- Training and Evaluation Functions ---
def train_ogp_np_task(model,orig_task_idx,assign_head_idx,opt_phi,cfg_obj):
    """Trains the OGP-NP model on a single task."""
    if not (0<=assign_head_idx<len(model.decoder_optimizers) and assign_head_idx < len(model.decoders)):
        raise ValueError(f"Train Error: Invalid head_idx {assign_head_idx} or optimizers/decoders not properly setup. Decoders: {len(model.decoders)}, Optimizers: {len(model.decoder_optimizers)}")
    
    optimizer_active_decoder = model.decoder_optimizers[assign_head_idx]
    x_train_task, y_train_task = task_data_train_global[orig_task_idx]
    if x_train_task.numel()==0:
        print(f"Task {orig_task_idx+1}: No training data available. Skipping training."); return

    train_dataset = TensorDataset(x_train_task, y_train_task)
    batch_size_train = min(cfg_obj.BATCH_SIZE_TRAIN, len(train_dataset))
    if batch_size_train == 0:
        print(f"Task {orig_task_idx+1}: Effective batch size is 0 (dataset too small). Skipping training."); return
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, 
                                  drop_last=(len(train_dataset)>batch_size_train and len(train_dataset)%batch_size_train!=0))

    model.train() 
    for i, decoder_module in enumerate(model.decoders):
        is_active_head = (i == assign_head_idx)
        decoder_module.train(mode=is_active_head)
        for param in decoder_module.parameters(): 
            param.requires_grad_(is_active_head)
    for param in model.get_phi_parameters(): 
        param.requires_grad_(True)

    print(f"Train Original Task {orig_task_idx+1} on Head {assign_head_idx} (Epochs: {cfg_obj.EPOCHS_PER_TASK})...")
    for epoch in range(cfg_obj.EPOCHS_PER_TASK):
        epoch_loss_sum, epoch_correct_preds, epoch_total_points, num_batches_processed = 0.0, 0, 0, 0
        for x_batch, y_labels_batch in train_dataloader:
            x_batch, y_labels_batch = x_batch.to(cfg_obj.DEVICE), y_labels_batch.to(cfg_obj.DEVICE)
            if x_batch.size(0)==0: continue 

            num_points_in_batch = x_batch.size(0)
            x_sequence_for_np = x_batch.reshape(1, num_points_in_batch, -1)
            y_sequence_labels_for_np = y_labels_batch.reshape(1, num_points_in_batch)

            x_ctx,y_ctx_onehot,_,x_trg,y_trg_lbl,is_valid_split = get_context_target_split(
                                                                x_sequence_for_np, y_sequence_labels_for_np,
                                                                model.y_dim_onehot,cfg_obj)
            if not is_valid_split: continue 

            opt_phi.zero_grad(set_to_none=True)
            optimizer_active_decoder.zero_grad(set_to_none=True)

            y_pred, z_mean, z_logvar = model(x_ctx,y_ctx_onehot,x_trg, assign_head_idx) 
            loss, _, _ = ogp_np_loss(y_pred,y_trg_lbl,z_mean,z_logvar,cfg_obj.KL_WEIGHT)

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: NaN/Inf loss encountered in epoch {epoch+1}. Skipping backward for this batch.")
                continue 

            loss.backward()
            if model.past_task_jacobians_stacked is not None and model.past_task_jacobians_stacked.numel()>0:
                project_gradients_ogp(model,cfg_obj)
            
            torch.nn.utils.clip_grad_norm_(model.get_phi_parameters(),1.0)
            torch.nn.utils.clip_grad_norm_(model.decoders[assign_head_idx].parameters(),1.0)

            opt_phi.step()
            optimizer_active_decoder.step()
            
            epoch_loss_sum += loss.item()
            _,predicted_classes = torch.max(y_pred.data,-1) 
            epoch_correct_preds += (predicted_classes.squeeze()==y_trg_lbl.squeeze()).sum().item()
            epoch_total_points += y_trg_lbl.numel()
            num_batches_processed += 1
        
        if num_batches_processed > 0:
            avg_epoch_loss = epoch_loss_sum / num_batches_processed
            epoch_accuracy = 100. * epoch_correct_preds / epoch_total_points if epoch_total_points > 0 else 0.0
            print(f"  Epoch {epoch+1}: AvgLoss={avg_epoch_loss:.4f}, Accuracy={epoch_accuracy:.2f}%")
        else:
            print(f"  Epoch {epoch+1}: No valid batches processed in this epoch.")


def evaluate_ogp_np_task(model,orig_task_idx_eval,cfg_obj):
    """Evaluates the OGP-NP model on a single task's test set."""
    x_test_task, y_test_task = task_data_test_global[orig_task_idx_eval]
    if x_test_task.numel()==0: 
        print(f"Eval Task {orig_task_idx_eval+1}: No test data. Accuracy = 0.0")
        return 0.0

    head_idx_for_eval = model.task_to_head_map.get(orig_task_idx_eval)
    if head_idx_for_eval is None or not (0<=head_idx_for_eval<len(model.decoders)):
        print(f"Eval Warning: No valid head assigned for task {orig_task_idx_eval+1} (Head: {head_idx_for_eval}, Decoders: {len(model.decoders)}). Accuracy will be 0.")
        return 0.0
    
    test_dataset = TensorDataset(x_test_task,y_test_task)
    batch_size_eval = min(cfg_obj.BATCH_SIZE_EVAL,len(test_dataset))
    if batch_size_eval == 0: 
        print(f"Eval Task {orig_task_idx_eval+1}: Effective batch size 0. Accuracy = 0.0")
        return 0.0
    
    test_dataloader = DataLoader(test_dataset,batch_size=batch_size_eval,shuffle=False,
                                 drop_last=(len(test_dataset)>batch_size_eval and len(test_dataset)%batch_size_eval!=0))
    
    model.eval() 
    total_correct_predictions, total_points_evaluated = 0,0

    with torch.no_grad(): 
        for batch_num,(x_batch,y_labels_batch) in enumerate(test_dataloader):
            if batch_num >= cfg_obj.NUM_EVAL_BATCHES: break 

            x_batch,y_labels_batch = x_batch.to(cfg_obj.DEVICE),y_labels_batch.to(cfg_obj.DEVICE)
            if x_batch.size(0)==0: continue

            num_points_in_batch = x_batch.size(0)
            x_sequence_for_np = x_batch.reshape(1, num_points_in_batch, -1)
            y_sequence_labels_for_np = y_labels_batch.reshape(1, num_points_in_batch)

            num_ctx_eval = min(cfg_obj.FIXED_EVAL_CONTEXT, num_points_in_batch -1 if num_points_in_batch > 1 else 0)
            num_ctx_eval = max(0, num_ctx_eval) 
            
            if (num_points_in_batch - num_ctx_eval) <= 0 : continue 

            ctx_indices_eval = torch.arange(num_ctx_eval, device=cfg_obj.DEVICE)
            trg_indices_eval = torch.arange(num_ctx_eval, num_points_in_batch, device=cfg_obj.DEVICE)

            x_ctx_eval, y_ctx_labels_eval = x_sequence_for_np[:,ctx_indices_eval,:], y_sequence_labels_for_np[:,ctx_indices_eval]
            x_trg_eval, y_trg_labels_eval = x_sequence_for_np[:,trg_indices_eval,:], y_sequence_labels_for_np[:,trg_indices_eval]

            y_ctx_onehot_eval = F.one_hot(y_ctx_labels_eval.long(),num_classes=model.y_dim_onehot).float() if num_ctx_eval > 0 else \
                                torch.empty(1,0,model.y_dim_onehot, device=cfg_obj.DEVICE,dtype=torch.float)
            if num_ctx_eval == 0: 
                x_ctx_eval = torch.empty(1,0,model.x_dim,device=cfg_obj.DEVICE,dtype=x_sequence_for_np.dtype)
            
            if x_trg_eval.size(1) == 0: continue 

            y_pred_eval,_,_ = model(x_ctx_eval,y_ctx_onehot_eval,x_trg_eval, head_idx_for_eval)
            _,predicted_classes_eval = torch.max(y_pred_eval.data,-1)

            total_correct_predictions += (predicted_classes_eval.squeeze()==y_trg_labels_eval.squeeze()).sum().item()
            total_points_evaluated += y_trg_labels_eval.numel()

    return 100.*total_correct_predictions/total_points_evaluated if total_points_evaluated>0 else 0.0


# --- Main Experiment Loop ---
def run_continual_learning_ogp_np_experiment(cfg_obj):
    """Runs the full continual learning experiment."""
    model = OGPNP(cfg_obj).to(cfg_obj.DEVICE)
    optimizer_phi = optim.Adam(model.get_phi_parameters(), lr=cfg_obj.LR)
    
    num_effective_tasks = cfg_obj.NUM_CL_TASKS_EFFECTIVE
    all_tasks_accuracy_matrix = np.full((num_effective_tasks, num_effective_tasks), np.nan)
    avg_accs_stream, active_heads_stream = [], []
    head_decision_log = [] # NEW: To log head decisions

    for current_task_idx in range(num_effective_tasks): 
        task_display_number = current_task_idx + 1
        print(f"\n--- Processing Original Task {task_display_number}/{num_effective_tasks} ({cfg_obj.EXPERIMENT_TYPE}) ---")
        model.eval() 
        
        x_train_current_task, y_train_current_task = task_data_train_global[current_task_idx]
        num_decision_context_points = min(x_train_current_task.size(0), cfg_obj.NUM_CONTEXT_JACOBIAN, 100) 
        
        assigned_head_idx = -1
        action_taken = "unknown_initial" # For head decision logging

        if not cfg_obj.FIXED_HEAD_PER_TASK and (num_decision_context_points == 0 or x_train_current_task.numel() == 0):
            print(f"Task {task_display_number}: Dynamic mode with no/small context data for Z. Using prior-like Z for head decision.")
            ctx_x_for_decision = torch.empty(1,0,model.x_dim, device=model.device_internal, 
                                             dtype=x_train_current_task.dtype if x_train_current_task.numel() > 0 else torch.float)
            ctx_y_onehot_for_decision = torch.empty(1,0,model.y_dim_onehot, device=model.device_internal, dtype=torch.float)
            assigned_head_idx, action_taken = model.decide_head_for_task(current_task_idx, ctx_x_for_decision, ctx_y_onehot_for_decision)
        else: 
            rand_indices_for_decision = torch.randperm(x_train_current_task.size(0))[:num_decision_context_points] if num_decision_context_points > 0 else torch.empty(0, dtype=torch.long)
            
            ctx_x_for_decision = x_train_current_task[rand_indices_for_decision].to(model.device_internal).unsqueeze(0) if num_decision_context_points > 0 else \
                                 torch.empty(1,0,model.x_dim, device=model.device_internal, dtype=x_train_current_task.dtype)
            ctx_y_labels_for_decision = y_train_current_task[rand_indices_for_decision].to(model.device_internal).unsqueeze(0) if num_decision_context_points > 0 else \
                                        torch.empty(1,0, dtype=torch.long, device=model.device_internal)
            ctx_y_onehot_for_decision = F.one_hot(ctx_y_labels_for_decision.long(),num_classes=model.y_dim_onehot).float() if num_decision_context_points > 0 else \
                                        torch.empty(1,0,model.y_dim_onehot, device=model.device_internal, dtype=torch.float)
            
            assigned_head_idx, action_taken = model.decide_head_for_task(current_task_idx, ctx_x_for_decision, ctx_y_onehot_for_decision)
        
        # Log head decision for this task
        head_decision_log.append({
            "task_index": current_task_idx,
            "task_display_number": task_display_number,
            "assigned_head_index": assigned_head_idx,
            "action": action_taken,
            "num_active_heads_after_decision": len(model.decoders)
        })
        active_heads_stream.append(len(model.decoders)) 
        print(f"Original Task {task_display_number} assigned to Head {assigned_head_idx} (Action: {action_taken}).")
        
        train_ogp_np_task(model, current_task_idx, assigned_head_idx, optimizer_phi, cfg_obj)
        model.eval()
        
        print(f"Collecting Jacobian for original task {task_display_number} (trained on head {assigned_head_idx})...")
        ctx_x_jac,ctx_y_onehot_jac,val_x_jac = get_ogp_data_for_jacobian(current_task_idx,cfg_obj)
        if ctx_x_jac is not None and val_x_jac is not None and val_x_jac.size(1)>0:
            jacobian_current_task = collect_jacobian_for_task_ogp_func(model,current_task_idx,
                                                               ctx_x_jac,ctx_y_onehot_jac,val_x_jac,cfg_obj)
            if jacobian_current_task is not None and jacobian_current_task.numel()>0:
                print(f"  Jacobian for Task {task_display_number} (head {assigned_head_idx}) shape: {jacobian_current_task.shape}")
                total_phi_dimension = sum(p.numel() for p in model.get_phi_parameters())
                if jacobian_current_task.shape[1]!=total_phi_dimension:
                    print(f"  OGP Jacobian Warning: Jacobian columns ({jacobian_current_task.shape[1]}) != Total phi params ({total_phi_dimension})")
                else: 
                    if model.past_task_jacobians_stacked is None or model.past_task_jacobians_stacked.numel()==0:
                        model.past_task_jacobians_stacked = jacobian_current_task
                    elif model.past_task_jacobians_stacked.shape[1] != jacobian_current_task.shape[1]: 
                        print("OGP Jacobian Warning: Phi dimension mismatch for Jacobian concatenation! Resetting J_stacked with current task's J.")
                        model.past_task_jacobians_stacked = jacobian_current_task
                    else:
                        model.past_task_jacobians_stacked = torch.cat([model.past_task_jacobians_stacked,jacobian_current_task],dim=0)
                if model.past_task_jacobians_stacked is not None:
                    print(f"  Total J_stacked (past tasks) shape: {model.past_task_jacobians_stacked.shape}")
            else:
                print(f"  Failed to collect/compute Jacobian for task {task_display_number}.")
        else:
            print(f"  Skipping Jacobian collection for task {task_display_number} (insufficient OGP data).")
        
        print(f"\n--- Evaluating after training Original Task {task_display_number} ---")
        for eval_task_idx in range(num_effective_tasks): 
            eval_task_display_number = eval_task_idx + 1
            if eval_task_idx <= current_task_idx: 
                accuracy = evaluate_ogp_np_task(model, eval_task_idx, cfg_obj)
                all_tasks_accuracy_matrix[current_task_idx, eval_task_idx] = accuracy
                head_used_for_eval = model.task_to_head_map.get(eval_task_idx,'N/A (Error)')
                print(f"  Accuracy on Original Task {eval_task_display_number}: {accuracy:.2f}% (using head {head_used_for_eval})")
        
        avg_accuracy_this_stage = np.nanmean(all_tasks_accuracy_matrix[current_task_idx, :current_task_idx+1])
        avg_accs_stream.append(avg_accuracy_this_stage)
        print(f"  Average accuracy (tasks 1 to {task_display_number}): {avg_accuracy_this_stage:.2f}% | Active Heads: {len(model.decoders)}")

    final_avg_accuracy = avg_accs_stream[-1] if avg_accs_stream else np.nan

    bwt = 0.0
    if num_effective_tasks > 1:
        for j in range(num_effective_tasks - 1): 
            acc_K_minus_1_j = all_tasks_accuracy_matrix[num_effective_tasks-1, j] 
            acc_j_j = all_tasks_accuracy_matrix[j, j]   
            if not np.isnan(acc_K_minus_1_j) and not np.isnan(acc_j_j): bwt += (acc_K_minus_1_j - acc_j_j)
        bwt /= (num_effective_tasks - 1)
    else: bwt = np.nan

    print(f"\nFinal Average Accuracy (at end of training): {final_avg_accuracy:.2f}%")
    print(f"Backward Transfer (BWT): {bwt:.4f}")
    
    return all_tasks_accuracy_matrix, avg_accs_stream, active_heads_stream, head_decision_log, bwt


# --- Main Execution Block ---
if __name__ == '__main__':
    parse_args_and_setup_config() 
    prepare_task_data() 
    
    mode_str_for_path = "FixedHead" if config.FIXED_HEAD_PER_TASK else "DynamicHead"
    
    if config.FIXED_HEAD_PER_TASK:
        results_dir_name = f"{config.EXPERIMENT_TYPE}_seed{config.SEED}_{mode_str_for_path}_{config.EPOCHS_PER_TASK}epochs"
    else: 
        results_dir_name = f"{config.EXPERIMENT_TYPE}_seed{config.SEED}_{mode_str_for_path}_{config.EPOCHS_PER_TASK}epochs_thresh{config.THRESHOLD}"
    
    os.makedirs(results_dir_name, exist_ok=True)
    print(f"Results will be saved in: {results_dir_name}")
    
    config_to_save = {k: str(v) if isinstance(v, torch.device) else v for k, v in vars(config).items()}
    with open(os.path.join(results_dir_name, 'config.json'), 'w') as f: 
        json.dump(config_to_save, f, indent=4)
    
    print(f"Starting OGP-NP CL Experiment: {config.EXPERIMENT_TYPE}, {config.NUM_CL_TASKS_EFFECTIVE} effective tasks.")
    print(f"Head allocation: {'Fixed per task' if config.FIXED_HEAD_PER_TASK else 'Dynamic based on Z-archetype KL (Threshold: ' + str(config.THRESHOLD) + ')'}")

    start_time = time.time()
    all_task_accs_matrix, avg_accs_stream, heads_log_stream, head_decision_details, bwt_final = run_continual_learning_ogp_np_experiment(config)
    end_time = time.time()
    experiment_duration_minutes = (end_time - start_time) / 60
    
    print(f"\nExperiment finished in {experiment_duration_minutes:.2f} minutes.")
    final_num_heads = heads_log_stream[-1] if heads_log_stream else 'N/A'
    if heads_log_stream: print(f"Final number of active heads: {final_num_heads}")
    
    results_data = {
        "config_used": config_to_save,
        "all_tasks_accuracy_matrix": all_task_accs_matrix.tolist() if all_task_accs_matrix is not None else [],
        "average_accuracies_over_time": avg_accs_stream, 
        "active_heads_over_time": heads_log_stream,     
        "head_decision_details_per_task": head_decision_details, # NEWLY ADDED
        "backward_transfer": bwt_final,
        "final_average_accuracy": avg_accs_stream[-1] if avg_accs_stream else np.nan,
        "final_num_heads": final_num_heads,
        "experiment_duration_minutes": experiment_duration_minutes
    }
    results_file_path = os.path.join(results_dir_name, 'results.pt') 
    torch.save(results_data, results_file_path)
    print(f"Numerical results saved to: {results_file_path}")

    print(f"\n--- OGP-NP Final Accuracies Table ({config.EXPERIMENT_TYPE}, {mode_str_for_path}) ---")
    num_effective_tasks_table = config.NUM_CL_TASKS_EFFECTIVE 
    header_string="Eval Task v | Train Stage -> |"
    for i in range(num_effective_tasks_table): header_string += f" {i+1:<7} |" 
    print(header_string)
    print("-" * len(header_string))

    for eval_task_idx_table in range(num_effective_tasks_table): 
        row_string=f"   Task {eval_task_idx_table+1:<2}    | "
        for training_stage_idx in range(num_effective_tasks_table): 
            accuracy_value = all_tasks_accuracy_matrix[training_stage_idx, eval_task_idx_table]
            if training_stage_idx < eval_task_idx_table: 
                row_string+="   -     | " 
            elif not np.isnan(accuracy_value):
                row_string+=f"{accuracy_value:7.2f} | "
            else: 
                row_string+="  N/A    | "
        print(row_string)

    if avg_accs_stream: print(f"\nFinal Average Accuracy (across tasks seen so far, at the very end): {avg_accs_stream[-1]:.2f}%")
    print(f"Final Backward Transfer (BWT): {bwt_final:.4f}")
    print(f"--- Experiment {results_dir_name} Complete ---")
