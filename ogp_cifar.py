import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Subset
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import numpy as np
import matplotlib.pyplot as plt # Keep for potential future use by user
import time
import argparse
from torch.func import functional_call, jacrev # For PyTorch 1.13+
# from functorch import jacrev, combine_state_for_ensemble # For older functorch
import os
import json
import random # For shuffling task order if desired later

# --- Configuration for Fixed Hyperparameters ---
class Config:
    # --- Data Dimensions & Dataset Specifics ---
    NUM_CLASSES = 10 # Default, will be updated by experiment type (10 for CIFAR10, 100 for CIFAR100, 200 for TinyImageNet)
    CNN_OUTPUT_DIM = 512 # Output dimension of the CNN feature extractor (e.g., for ResNet9)
    # X_DIM will be set to CNN_OUTPUT_DIM for image tasks, or 784 for MNIST
    X_DIM = CNN_OUTPUT_DIM 
    Y_DIM_ONEHOT = NUM_CLASSES 
    Y_DIM_OUT = NUM_CLASSES    

    # --- Neural Process Architecture Dimensions ---
    R_DIM = 128  # Dimension of the deterministic representation r_i from NPEncoder
    Z_DIM = 64  # Dimension of the latent variable z from LatentEncoder
    ENC_HIDDEN_DIM = 128  # Hidden dimension for NPEncoder and LatentEncoder MLPs
    DEC_HIDDEN_DIM = 128  # Hidden dimension for NPDecoder MLP

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
    OPM_PROJ_DIM = 200  # If J_old has more rows (constraints) than this, project J_old to this dimension for OGP

    # --- Evaluation Hyperparameters ---
    NUM_EVAL_BATCHES = 50  # Max number of batches to use during evaluation per task (for speed)
    FIXED_EVAL_CONTEXT = 30  # Number of context points to use from each batch during evaluation

    # --- Experiment Setup ---
    EXPERIMENT_TYPE = 'SplitCIFAR10'  # Type of experiment
    NUM_TASKS = 5  # Default, will be derived for split tasks. Used for Permuted/Rotated MNIST if selected.
    
    # MNIST Specific (if used)
    MNIST_ROTATED_MAX_ANGLE = 90.0
    
    # CIFAR-10 Specific
    CIFAR10_CLASSES_PER_TASK = 2 # For SplitCIFAR10 (e.g., 5 tasks of 2 classes)

    # CIFAR-100 Specific
    CIFAR100_CLASSES_PER_TASK = 10 # For SplitCIFAR100 (e.g., 10 tasks of 10 classes)
    CIFAR100_SUPERCLASSES = False # If true, could split by 20 superclasses

    # Tiny ImageNet Specific (Placeholder)
    TINYIMGNET_TOTAL_CLASSES = 200
    TINYIMGNET_CLASSES_PER_TASK = 20 # For SplitTinyImageNet (e.g., 10 tasks of 20 classes)
    TINYIMGNET_IMG_SIZE = 64 # Tiny ImageNet images are 64x64

    DATA_ROOT = './data'  # Root directory for storing datasets
    SEED = 42  # Random seed for reproducibility
    FIXED_HEAD_PER_TASK = False  # If True, task N uses head N, overriding dynamic allocation
    DEVICE = torch.device("cpu") # Will be updated based on CUDA availability


config = Config() # Global config object
device = torch.device("cpu") # Global device, will be updated
task_data_train_global = [] # Global list to store training data for each task
task_data_test_global = []  # Global list to store test data for each task

# --- CNN Backbone (SmallResNet for CIFAR-like images) ---
def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
              nn.BatchNorm2d(out_channels),
              nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

class SmallResNet(nn.Module):
    def __init__(self, in_channels=3, output_dim=config.CNN_OUTPUT_DIM):
        super().__init__()
        self.conv1 = conv_block(in_channels, 64)
        self.conv2 = conv_block(64, 128, pool=True)
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))
        
        self.conv3 = conv_block(128, 256, pool=True)
        self.conv4 = conv_block(256, 512, pool=True)
        self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))
        
        self.pool = nn.AdaptiveMaxPool2d(1) # Global Max Pooling
        self.fc = nn.Linear(512, output_dim) if output_dim is not None and output_dim > 0 else nn.Identity() # Output features

    def forward(self, xb): # xb: (B, C, H, W)
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.pool(out)    # (B, 512, 1, 1)
        out = out.view(out.size(0), -1) # (B, 512)
        out = self.fc(out) # (B, output_dim) or (B, 512) if fc is Identity
        return out

def parse_args_and_setup_config():
    """Parses command-line arguments and updates the global config object and device."""
    global config, device
    parser = argparse.ArgumentParser(description="OGP-NP for Continual Learning on Image Datasets.")
    parser.add_argument('--experiment_type', type=str, default=config.EXPERIMENT_TYPE,
                        choices=['PermutedMNIST', 'SplitCIFAR10', 'SplitCIFAR100', 'TinyImageNet', 'RotatedMNIST'],
                        help='Type of experiment.')
    # num_tasks is mainly for Permuted/Rotated MNIST. Split tasks derive their own.
    parser.add_argument('--num_tasks', type=int, default=config.NUM_TASKS, help='Number of tasks for relevant experiments.')
    parser.add_argument('--epochs_per_task', type=int, default=config.EPOCHS_PER_TASK, help='Epochs per task.')
    parser.add_argument('--seed', type=int, default=config.SEED, help='Random seed.')
    parser.add_argument('--fixed_head_per_task', action='store_true', default=config.FIXED_HEAD_PER_TASK,
                        help='If set, task N uses head N, overriding dynamic head allocation.')
    parser.add_argument('--threshold', type=float, default=config.THRESHOLD,
                        help='Symmetric KL Divergence threshold for dynamic head spawning.')
    cli_args = parser.parse_args()

    config.SEED = cli_args.seed
    config.NUM_TASKS = cli_args.num_tasks 
    config.EPOCHS_PER_TASK = cli_args.epochs_per_task
    config.EXPERIMENT_TYPE = cli_args.experiment_type
    config.FIXED_HEAD_PER_TASK = cli_args.fixed_head_per_task
    config.THRESHOLD = cli_args.threshold

    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    random.seed(config.SEED) # For potential task order shuffling

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.DEVICE = device
    
    # Update NUM_CLASSES and related dims based on experiment type
    if config.EXPERIMENT_TYPE == 'SplitCIFAR10':
        config.NUM_CLASSES = 10
    elif config.EXPERIMENT_TYPE == 'SplitCIFAR100':
        config.NUM_CLASSES = 100
    elif config.EXPERIMENT_TYPE == 'TinyImageNet':
        config.NUM_CLASSES = config.TINYIMGNET_TOTAL_CLASSES
    elif "MNIST" in config.EXPERIMENT_TYPE: # PermutedMNIST or RotatedMNIST
        config.NUM_CLASSES = 10
        config.X_DIM = 784 # MNIST uses flattened pixels directly for NPEncoder if no CNN
    
    config.Y_DIM_ONEHOT = config.NUM_CLASSES
    config.Y_DIM_OUT = config.NUM_CLASSES
    if "MNIST" not in config.EXPERIMENT_TYPE: # For CIFAR/TinyImageNet, X_DIM is CNN output
        config.X_DIM = config.CNN_OUTPUT_DIM

    print(f"Global device set to: {device}")
    print(f"Running with configuration: {vars(config)}")


def prepare_task_data():
    """Loads and preprocesses task data based on the experiment type."""
    global task_data_train_global, task_data_test_global, config
    task_data_train_global.clear(); task_data_test_global.clear()
    
    actual_num_cl_tasks = config.NUM_TASKS # Default, might be overridden

    # --- CIFAR-10 ---
    if config.EXPERIMENT_TYPE == "SplitCIFAR10":
        print(f"Loading Split CIFAR-10 ({config.CIFAR10_CLASSES_PER_TASK} classes per task)...")
        cifar10_mean = (0.4914, 0.4822, 0.4465)
        cifar10_std = (0.2023, 0.1994, 0.2010)
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(cifar10_mean, cifar10_std),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(cifar10_mean, cifar10_std),
        ])
        
        train_dataset_full = torchvision.datasets.CIFAR10(root=config.DATA_ROOT, train=True, download=True, transform=transform_train)
        test_dataset_full = torchvision.datasets.CIFAR10(root=config.DATA_ROOT, train=False, download=True, transform=transform_test)
        
        actual_num_cl_tasks = config.NUM_CLASSES // config.CIFAR10_CLASSES_PER_TASK
        print(f"  Derived number of CL tasks for SplitCIFAR10: {actual_num_cl_tasks}")

        all_labels = list(range(config.NUM_CLASSES))
        # random.shuffle(all_labels) # Optional: shuffle class order before splitting for more variety

        for i in range(actual_num_cl_tasks):
            task_labels = all_labels[i * config.CIFAR10_CLASSES_PER_TASK : (i + 1) * config.CIFAR10_CLASSES_PER_TASK]
            
            train_indices = [idx for idx, target in enumerate(train_dataset_full.targets) if target in task_labels]
            test_indices = [idx for idx, target in enumerate(test_dataset_full.targets) if target in task_labels]
            
            # Create Subset datasets
            task_train_data = Subset(train_dataset_full, train_indices)
            task_test_data = Subset(test_dataset_full, test_indices)
            
            # Convert Subset to Tensors for global list (or adapt DataLoader in train/eval)
            # For simplicity here, we'll convert to tensors. In practice, passing Subsets to DataLoader is better.
            train_x = torch.stack([task_train_data[j][0] for j in range(len(task_train_data))])
            train_y = torch.tensor([task_train_data[j][1] for j in range(len(task_train_data))])
            test_x = torch.stack([task_test_data[j][0] for j in range(len(task_test_data))])
            test_y = torch.tensor([task_test_data[j][1] for j in range(len(task_test_data))])

            task_data_train_global.append((train_x, train_y))
            task_data_test_global.append((test_x, test_y))
            print(f"  SplitCIFAR10 Task {i+1}: labels {task_labels}, train {len(train_x)}, test {len(test_x)}")

    # --- CIFAR-100 ---
    elif config.EXPERIMENT_TYPE == "SplitCIFAR100":
        print(f"Loading Split CIFAR-100 ({config.CIFAR100_CLASSES_PER_TASK} classes per task)...")
        cifar100_mean = (0.5071, 0.4867, 0.4408)
        cifar100_std = (0.2675, 0.2565, 0.2761)
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(cifar100_mean, cifar100_std),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(cifar100_mean, cifar100_std),
        ])
        
        train_dataset_full = torchvision.datasets.CIFAR100(root=config.DATA_ROOT, train=True, download=True, transform=transform_train)
        test_dataset_full = torchvision.datasets.CIFAR100(root=config.DATA_ROOT, train=False, download=True, transform=transform_test)
        
        actual_num_cl_tasks = config.NUM_CLASSES // config.CIFAR100_CLASSES_PER_TASK
        print(f"  Derived number of CL tasks for SplitCIFAR100: {actual_num_cl_tasks}")

        all_labels = list(range(config.NUM_CLASSES))
        # random.shuffle(all_labels) # Optional

        for i in range(actual_num_cl_tasks):
            task_labels = all_labels[i * config.CIFAR100_CLASSES_PER_TASK : (i + 1) * config.CIFAR100_CLASSES_PER_TASK]
            
            train_indices = [idx for idx, target in enumerate(train_dataset_full.targets) if target in task_labels]
            test_indices = [idx for idx, target in enumerate(test_dataset_full.targets) if target in task_labels]
            
            train_x = torch.stack([train_dataset_full[j][0] for j in train_indices])
            train_y = torch.tensor([train_dataset_full.targets[j] for j in train_indices])
            test_x = torch.stack([test_dataset_full[j][0] for j in test_indices])
            test_y = torch.tensor([test_dataset_full.targets[j] for j in test_indices])

            task_data_train_global.append((train_x, train_y))
            task_data_test_global.append((test_x, test_y))
            print(f"  SplitCIFAR100 Task {i+1}: labels {task_labels}, train {len(train_x)}, test {len(test_x)}")
            
    # --- Tiny ImageNet (Placeholder with Dummy Data) ---
    elif config.EXPERIMENT_TYPE == "TinyImageNet":
        print(f"Loading Split Tiny ImageNet (Placeholder - DUMMY DATA) ({config.TINYIMGNET_CLASSES_PER_TASK} classes per task)...")
        actual_num_cl_tasks = config.TINYIMGNET_TOTAL_CLASSES // config.TINYIMGNET_CLASSES_PER_TASK
        print(f"  Derived number of CL tasks for TinyImageNet: {actual_num_cl_tasks}")
        print("  WARNING: Using DUMMY DATA for TinyImageNet. Implement actual data loading.")

        img_size = config.TINYIMGNET_IMG_SIZE
        num_dummy_samples_train = 500 # Per task
        num_dummy_samples_test = 100  # Per task

        all_labels = list(range(config.TINYIMGNET_TOTAL_CLASSES))
        # random.shuffle(all_labels)

        for i in range(actual_num_cl_tasks):
            task_labels = all_labels[i * config.TINYIMGNET_CLASSES_PER_TASK : (i + 1) * config.TINYIMGNET_CLASSES_PER_TASK]
            
            # Create dummy data
            dummy_train_x = torch.randn(num_dummy_samples_train, 3, img_size, img_size)
            # Assign labels from the current task_labels subset
            dummy_train_y = torch.tensor(np.random.choice(task_labels, num_dummy_samples_train)) 
            
            dummy_test_x = torch.randn(num_dummy_samples_test, 3, img_size, img_size)
            dummy_test_y = torch.tensor(np.random.choice(task_labels, num_dummy_samples_test))

            task_data_train_global.append((dummy_train_x, dummy_train_y))
            task_data_test_global.append((dummy_test_x, dummy_test_y))
            print(f"  TinyImageNet Task {i+1} (Dummy): labels {task_labels}, train {len(dummy_train_x)}, test {len(dummy_test_x)}")
            
    # --- Permuted MNIST / Rotated MNIST (Legacy, if selected) ---
    elif "MNIST" in config.EXPERIMENT_TYPE:
        print(f"Loading {config.EXPERIMENT_TYPE}...")
        # MNIST uses flattened pixels, X_DIM is 784, NUM_CLASSES is 10
        # These are set in parse_args_and_setup_config based on MNIST selection
        mnist_mean, mnist_std = (0.1307,), (0.3081,)
        normalize_transform = transforms.Normalize(mnist_mean, mnist_std)
        full_transform = transforms.Compose([transforms.ToTensor(), normalize_transform])
        _mnist_train_full = torchvision.datasets.MNIST(root=config.DATA_ROOT, train=True, download=True, transform=full_transform)
        _mnist_test_full = torchvision.datasets.MNIST(root=config.DATA_ROOT, train=False, download=True, transform=full_transform)
        
        x_train_base = torch.stack([img.view(-1) for img, _ in _mnist_train_full])
        y_train_base = torch.tensor([target for _, target in _mnist_train_full])
        x_test_base = torch.stack([img.view(-1) for img, _ in _mnist_test_full])
        y_test_base = torch.tensor([target for _, target in _mnist_test_full])

        if config.EXPERIMENT_TYPE == "PermutedMNIST":
            task_data_train_global.append((x_train_base, y_train_base))
            task_data_test_global.append((x_test_base, y_test_base))
            print(f"  PermutedMNIST Task 1 (Original): train {len(x_train_base)}, test {len(x_test_base)}")
            for i in range(1, actual_num_cl_tasks): 
                perm = torch.randperm(config.X_DIM) # Use torch.randperm for tensor permutation
                task_data_train_global.append((x_train_base[:, perm], y_train_base))
                task_data_test_global.append((x_test_base[:, perm], y_test_base))
                print(f"  PermutedMNIST Task {i+1} (Permuted): train {len(x_train_base)}, test {len(x_test_base)}")
        elif config.EXPERIMENT_TYPE == "RotatedMNIST":
            # This part needs TF.rotate which expects CHW, so we'd need to reshape before rotate
            # For simplicity, this part is kept minimal as focus is on CIFAR
            print("  RotatedMNIST for raw pixels needs careful reshaping for TF.rotate. Using original for now.")
            # Placeholder: just adds original MNIST multiple times for RotatedMNIST if selected
            # Proper RotatedMNIST would require reshaping to (C,H,W), rotating, then flattening.
            for i in range(actual_num_cl_tasks):
                task_data_train_global.append((x_train_base.clone(), y_train_base.clone()))
                task_data_test_global.append((x_test_base.clone(), y_test_base.clone()))
                print(f"  RotatedMNIST Task {i+1} (Placeholder - Original Data): train {len(x_train_base)}, test {len(x_test_base)}")


    else: 
        raise ValueError(f"Unsupported experiment_type: {config.EXPERIMENT_TYPE}")
    
    config.NUM_CL_TASKS_EFFECTIVE = actual_num_cl_tasks


# --- Neural Process Model Components (NPEncoder, LatentEncoder, NPDecoder are mostly the same) ---
# (Definitions of NPEncoder, LatentEncoder, NPDecoder, KL utils are the same as in ogp_np_experiment_py_v3)
# For brevity, I'll skip pasting them again here, assuming they are available from the previous version.
# Ensure their input/output dimensions match the feature dimensions if a CNN is used.
# The NPEncoder's input_dim will be config.X_DIM (which is CNN_OUTPUT_DIM for image tasks)
# The NPDecoder's x_dim input will also be config.X_DIM (feature dimension)

# --- OGP-NP Model ---
class OGPNP(nn.Module):
    """Orthogonal Gradient Projection Neural Process model with CNN backbone."""
    def __init__(self, cfg_obj):
        super(OGPNP, self).__init__()
        self.cfg = cfg_obj
        self.is_image_task = "MNIST" not in cfg_obj.EXPERIMENT_TYPE

        if self.is_image_task:
            self.cnn_feature_extractor = SmallResNet(in_channels=3, output_dim=cfg_obj.CNN_OUTPUT_DIM)
            # X_DIM is the feature dimension from CNN
            self.feature_dim_internal = cfg_obj.CNN_OUTPUT_DIM 
        else: # MNIST like tasks, no CNN, features are raw pixels (flattened)
            self.cnn_feature_extractor = nn.Identity() # No-op for MNIST
            self.feature_dim_internal = cfg_obj.X_DIM # 784 for MNIST

        self.y_dim_onehot = cfg_obj.Y_DIM_ONEHOT
        self.y_dim_out = cfg_obj.Y_DIM_OUT
        self.r_dim = cfg_obj.R_DIM
        self.z_dim = cfg_obj.Z_DIM
        self.enc_hidden_dim = cfg_obj.ENC_HIDDEN_DIM
        self.dec_hidden_dim = cfg_obj.DEC_HIDDEN_DIM
        
        # NP components operate on features
        self.xy_encoder = NPEncoder(self.feature_dim_internal, self.y_dim_onehot, self.enc_hidden_dim, self.r_dim)
        self.latent_encoder = LatentEncoder(self.r_dim, self.z_dim)
        
        # Phi modules include CNN (if used) and NP encoders
        if self.is_image_task:
            self.phi_modules = nn.ModuleList([self.cnn_feature_extractor, self.xy_encoder, self.latent_encoder])
        else:
            self.phi_modules = nn.ModuleList([self.xy_encoder, self.latent_encoder]) # MNIST has no separate CNN in phi
        
        self.decoders = nn.ModuleList()
        self.decoder_optimizers = []
        self.head_archetype_z_params = [] 
        self.head_task_counts = [] 
        
        self.task_to_head_map = {} 
        self.past_task_jacobians_stacked = None
        self.device_internal = cfg_obj.DEVICE

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        new_device_target = None
        if args and isinstance(args[0], (str, torch.device)): 
            new_device_target = torch.device(args[0])
        elif kwargs and 'device' in kwargs: 
            new_device_target = torch.device(kwargs['device'])
        if new_device_target is None: 
            try: new_device_target = next(self.parameters()).device
            except StopIteration: pass 
        if new_device_target and self.device_internal != new_device_target:
            self.device_internal = new_device_target
            self.head_archetype_z_params = [(m.to(self.device_internal), lv.to(self.device_internal))
                                            for m, lv in self.head_archetype_z_params
                                            if isinstance(m, torch.Tensor) and isinstance(lv, torch.Tensor)]
        return self

    def get_phi_parameters(self):
        return [p for module in self.phi_modules for p in module.parameters()]

    def aggregate(self, r_i): 
        return torch.mean(r_i, dim=1) if r_i.size(1) > 0 else torch.zeros(r_i.size(0), self.r_dim, device=r_i.device)

    def reparameterize(self, z_mean, z_logvar): 
        std_dev = torch.exp(0.5 * z_logvar)
        epsilon = torch.randn_like(std_dev) 
        return z_mean + epsilon * std_dev 

    @torch.no_grad()
    def get_task_z_distribution_params(self, x_ctx_images, y_ctx_onehot): # x_ctx_images are raw images
        """Computes Z parameters. Input x_ctx_images are raw images (B, N, C, H, W) or (B, N, D) for MNIST."""
        original_modes = [module.training for module in self.phi_modules]
        for module in self.phi_modules: module.eval()

        batch_size_ctx, num_ctx_points, *_ = x_ctx_images.shape
        
        if num_ctx_points == 0:
            z_mean = torch.zeros(batch_size_ctx, self.z_dim, device=self.device_internal)
            z_logvar = torch.zeros(batch_size_ctx, self.z_dim, device=self.device_internal)
        else:
            if self.is_image_task: # Reshape for CNN: (B*N, C, H, W)
                img_c, img_h, img_w = x_ctx_images.shape[-3:]
                x_ctx_features = self.cnn_feature_extractor(x_ctx_images.reshape(-1, img_c, img_h, img_w))
                x_ctx_features = x_ctx_features.view(batch_size_ctx, num_ctx_points, -1) # Reshape back: (B, N, FeatDim)
            else: # MNIST: x_ctx_images is already (B, N, PixelDim)
                x_ctx_features = x_ctx_images 
            
            r_i = self.xy_encoder(x_ctx_features, y_ctx_onehot)
            r_agg = self.aggregate(r_i)
            z_mean, z_logvar = self.latent_encoder(r_agg)
        
        for i, module in enumerate(self.phi_modules): module.train(original_modes[i])
        return z_mean, z_logvar

    def _spawn_new_head(self, archetypal_zm_for_new_head, archetypal_zlv_for_new_head):
        # NPDecoder operates on features
        new_decoder = NPDecoder(self.feature_dim_internal, self.z_dim, self.dec_hidden_dim, self.y_dim_out).to(self.device_internal)
        self.decoders.append(new_decoder)
        new_head_idx = len(self.decoders) - 1
        optimizer_decoder = optim.Adam(new_decoder.parameters(), lr=self.cfg.LR)
        self.decoder_optimizers.append(optimizer_decoder)
        self.head_archetype_z_params.append(
            (archetypal_zm_for_new_head.detach().clone(), archetypal_zlv_for_new_head.detach().clone())
        )
        self.head_task_counts.append(1) 
        return new_head_idx

    def decide_head_for_task(self, orig_task_idx, task_ctx_images, task_ctx_y_onehot): # Takes raw images
        assigned_head_idx = -1
        action_string = "unknown"

        if self.cfg.FIXED_HEAD_PER_TASK:
            assigned_head_idx = orig_task_idx 
            action_string = "fixed_assignment_ensured_exists"
            while len(self.decoders) <= assigned_head_idx:
                dummy_zm = torch.zeros(self.z_dim, device=self.device_internal)
                dummy_zlv = torch.zeros(self.z_dim, device=self.device_internal)
                self._spawn_new_head(dummy_zm, dummy_zlv)
            print(f"Task {orig_task_idx+1}: Fixed head assignment to Head {assigned_head_idx}.")
        else: 
            current_task_azm_single, current_task_azlv_single = None, None
            # task_ctx_images has shape (B, N, C, H, W) or (B, N, D)
            if task_ctx_images.size(1) >= self.cfg.MIN_SAMPLES_FOR_PROTOTYPE: # Check num_points
                collected_z_means, collected_z_logvars = [], []
                for _ in range(self.cfg.NUM_Z_COLLECTIONS_FOR_PROTOTYPE):
                    # get_task_z_distribution_params expects raw images
                    zm_batch, zlv_batch = self.get_task_z_distribution_params(task_ctx_images, task_ctx_y_onehot)
                    collected_z_means.append(zm_batch.squeeze(0)) 
                    collected_z_logvars.append(zlv_batch.squeeze(0))
                if collected_z_means: 
                    current_task_azm_single = torch.stack(collected_z_means).mean(dim=0)
                    current_task_azlv_single = torch.stack(collected_z_logvars).mean(dim=0)
                else: 
                    current_task_azm_single = torch.zeros(self.z_dim, device=self.device_internal)
                    current_task_azlv_single = torch.zeros(self.z_dim, device=self.device_internal)
            else: 
                print(f"Task {orig_task_idx+1}: Context size {task_ctx_images.size(1)} < min_samples {self.cfg.MIN_SAMPLES_FOR_PROTOTYPE}. Using prior-like Z for head decision.")
                current_task_azm_single = torch.zeros(self.z_dim, device=self.device_internal)
                current_task_azlv_single = torch.zeros(self.z_dim, device=self.device_internal)
            
            current_task_azm_single = current_task_azm_single.detach().clone().to(self.device_internal)
            current_task_azlv_single = current_task_azlv_single.detach().clone().to(self.device_internal)

            if not self.decoders: 
                assigned_head_idx = self._spawn_new_head(current_task_azm_single, current_task_azlv_single)
                action_string = "dynamic_spawned_first_head"
                print(f"Task {orig_task_idx+1}: No heads exist (dynamic). Spawning head {assigned_head_idx} with current task's Z as archetype.")
            else: 
                best_matching_head_idx = -1; min_kl_divergence = float('inf')
                for head_idx_iter in range(len(self.decoders)):
                    archetype_zm, archetype_zlv = self.head_archetype_z_params[head_idx_iter]
                    kl_div_tensor = symmetric_kl_divergence(
                        current_task_azm_single.unsqueeze(0), current_task_azlv_single.unsqueeze(0),
                        archetype_zm.unsqueeze(0), archetype_zlv.unsqueeze(0))
                    kl_div = kl_div_tensor.item() 
                    if kl_div < min_kl_divergence:
                        min_kl_divergence = kl_div; best_matching_head_idx = head_idx_iter
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

    def forward(self, x_context_images, y_context_onehot, x_target_images, head_idx):
        """Forward pass. Inputs x_context_images and x_target_images are raw images."""
        if not (0 <= head_idx < len(self.decoders)):
             raise ValueError(f"Invalid head_idx {head_idx} for forward. Decoders: {len(self.decoders)}, requested: {head_idx}")
        
        batch_size_ctx, num_ctx_points, *_ = x_context_images.shape
        batch_size_trg, num_trg_points, *_ = x_target_images.shape

        # 1. Extract features from context images
        if num_ctx_points > 0:
            if self.is_image_task:
                img_c, img_h, img_w = x_context_images.shape[-3:]
                x_context_features = self.cnn_feature_extractor(x_context_images.reshape(-1, img_c, img_h, img_w))
                x_context_features = x_context_features.view(batch_size_ctx, num_ctx_points, -1)
            else:
                x_context_features = x_context_images
        else: # Handle empty context
            x_context_features = torch.empty(batch_size_ctx, 0, self.feature_dim_internal, device=self.device_internal)
            y_context_onehot = torch.empty(batch_size_ctx, 0, self.y_dim_onehot, device=self.device_internal)


        # 2. Encode context to get Z
        r_i_ctx = self.xy_encoder(x_context_features, y_context_onehot)
        r_agg = self.aggregate(r_i_ctx)
        z_mean, z_logvar = self.latent_encoder(r_agg)
        z_sample = self.reparameterize(z_mean, z_logvar)

        # 3. Extract features from target images
        if num_trg_points > 0:
            if self.is_image_task:
                img_c, img_h, img_w = x_target_images.shape[-3:]
                x_target_features = self.cnn_feature_extractor(x_target_images.reshape(-1, img_c, img_h, img_w))
                x_target_features = x_target_features.view(batch_size_trg, num_trg_points, -1)
            else:
                x_target_features = x_target_images
        else: # Handle empty target (should not happen if split is valid)
             x_target_features = torch.empty(batch_size_trg, 0, self.feature_dim_internal, device=self.device_internal)


        # 4. Decode for target points using the specified head
        # NPDecoder expects features as its x_target input
        return self.decoders[head_idx](z_sample, x_target_features), z_mean, z_logvar


# --- Loss Function (same as before) ---
def ogp_np_loss(y_pred,y_trg_lbl,zm,zlv,kl_w):
    y_trg_lbl=y_trg_lbl.long() 
    ce_loss = F.cross_entropy(y_pred.reshape(-1,y_pred.size(-1)),y_trg_lbl.reshape(-1),reduction='mean')
    kl_individual_samples = -0.5 * torch.sum(1 + zlv - zm.pow(2) - zlv.exp(), dim=1) 
    kl_term = torch.mean(kl_individual_samples) 
    total_loss = ce_loss + kl_w * kl_term
    return total_loss, ce_loss, kl_term

# --- Data Utilities (get_context_target_split needs to handle image shapes if not flattened in global list) ---
def get_context_target_split(x_seq_batch_raw_images, y_seq_labels_batch, y_dim_onehot_cfg, cfg_obj):
    """Splits a batch of sequences (raw images) into context and target sets for NP training."""
    # x_seq_batch_raw_images: (1, num_total_points, C, H, W) or (1, num_total_points, D) for MNIST
    # y_seq_labels_batch: (1, num_total_points)
    _, total_points_in_sequence, *_ = x_seq_batch_raw_images.shape # Get num_points
    
    if total_points_in_sequence == 0: 
        empty_x_shape = list(x_seq_batch_raw_images.shape)
        empty_x_shape[1] = 0 # num_points = 0
        return (torch.empty(empty_x_shape, device=x_seq_batch_raw_images.device, dtype=x_seq_batch_raw_images.dtype),
                torch.empty_like(x_seq_batch_raw_images.expand(-1, 0, y_dim_onehot_cfg)), 
                torch.empty_like(y_seq_labels_batch.expand(-1, 0)), 
                torch.empty(empty_x_shape, device=x_seq_batch_raw_images.device, dtype=x_seq_batch_raw_images.dtype),
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
    
    indices = torch.randperm(total_points_in_sequence, device=x_seq_batch_raw_images.device)
    ctx_indices, target_indices = indices[:num_context], indices[num_context:]
    
    if target_indices.numel() == 0 and total_points_in_sequence > 0: 
        if num_context > 0 : 
            target_indices = ctx_indices[-1:].clone() 
            ctx_indices = ctx_indices[:-1] 
            num_context -=1
        else: 
            return get_context_target_split(x_seq_batch_raw_images,y_seq_labels_batch,y_dim_onehot_cfg,cfg_obj)

    # Select raw images for context and target
    x_ctx_raw, y_ctx_lbl = x_seq_batch_raw_images[:,ctx_indices,...], y_seq_labels_batch[:,ctx_indices]
    x_trg_raw, y_trg_lbl = x_seq_batch_raw_images[:,target_indices,...], y_seq_labels_batch[:,target_indices]
    
    y_ctx_onehot = F.one_hot(y_ctx_lbl.long(), num_classes=y_dim_onehot_cfg).float() if num_context > 0 else \
                   torch.empty(1,0,y_dim_onehot_cfg, device=x_seq_batch_raw_images.device, dtype=torch.float)
    
    valid_split = (x_trg_raw.size(1) > 0) 
    return x_ctx_raw,y_ctx_onehot,y_ctx_lbl,x_trg_raw,y_trg_lbl,valid_split


@torch.no_grad()
def get_ogp_data_for_jacobian(task_id,cfg_obj):
    """Prepares context and target data (raw images) for Jacobian calculation."""
    x_cpu_raw_images,y_cpu_labels=task_data_train_global[task_id] 
    x_cpu_raw_images,y_cpu_labels=x_cpu_raw_images.cpu(),y_cpu_labels.cpu()
    total_samples=x_cpu_raw_images.size(0)
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
    
    # Get raw images and move to device, add batch dim
    ctx_x_raw = x_cpu_raw_images[ctx_indices].to(cfg_obj.DEVICE).unsqueeze(0) 
    ctx_y_lbl = y_cpu_labels[ctx_indices].to(cfg_obj.DEVICE).unsqueeze(0) 
    ctx_y_onehot = F.one_hot(ctx_y_lbl.long(), num_classes=cfg_obj.Y_DIM_ONEHOT).float().to(cfg_obj.DEVICE)
    trg_x_raw = x_cpu_raw_images[val_indices].to(cfg_obj.DEVICE).unsqueeze(0) 

    if num_ctx_jac == 0: # Handle empty context
        empty_x_shape = list(x_cpu_raw_images.shape) # (C, H, W) or (D)
        empty_x_shape.insert(0,0) # (0, C, H, W) or (0,D)
        empty_x_shape.insert(0,1) # (1, 0, C, H, W) or (1,0,D)
        ctx_x_raw = torch.empty(empty_x_shape, device=cfg_obj.DEVICE, dtype=x_cpu_raw_images.dtype)
        ctx_y_onehot = torch.empty(1, 0, cfg_obj.Y_DIM_ONEHOT, device=cfg_obj.DEVICE, dtype=torch.float)
    
    return ctx_x_raw,ctx_y_onehot,trg_x_raw # Return raw images


# --- OGP Jacobian Calculation and Projection (collect_jacobian_for_task_ogp_func needs careful update) ---
def collect_jacobian_for_task_ogp_func(model,orig_task_id,ctx_x_raw_i,ctx_y_oh_i,trg_x_raw_val_i,cfg_obj):
    """Collects Jacobian. Inputs are raw images."""
    model_device = model.device_internal
    ctx_x_raw_i = ctx_x_raw_i.to(model_device); ctx_y_oh_i = ctx_y_oh_i.to(model_device); trg_x_raw_val_i = trg_x_raw_val_i.to(model_device)

    phi_param_tensors, phi_param_names_global = [], []
    cnn_param_names_local, xy_encoder_param_names_local, latent_encoder_param_names_local = [], [], []
    num_phi_cnn, num_phi_xy_encoder = 0, 0 # Store lengths for indexing

    # Gather CNN parameters (if applicable)
    if model.is_image_task:
        for name, param in sorted(model.cnn_feature_extractor.named_parameters(), key=lambda item: item[0]):
            phi_param_tensors.append(param); phi_param_names_global.append(f"cnn.{name}"); cnn_param_names_local.append(name)
        num_phi_cnn = len(cnn_param_names_local)
        buffers_cnn = {name: buffer.to(model_device) for name, buffer in model.cnn_feature_extractor.named_buffers()}
    else: buffers_cnn = {}


    # Gather XY encoder parameters
    for name, param in sorted(model.xy_encoder.named_parameters(), key=lambda item: item[0]):
        phi_param_tensors.append(param); phi_param_names_global.append(f"xy_encoder.{name}"); xy_encoder_param_names_local.append(name)
    num_phi_xy_encoder = len(xy_encoder_param_names_local)
    buffers_xy_encoder = {name: buffer.to(model_device) for name, buffer in model.xy_encoder.named_buffers()}

    # Gather Latent encoder parameters
    for name, param in sorted(model.latent_encoder.named_parameters(), key=lambda item: item[0]):
        phi_param_tensors.append(param); phi_param_names_global.append(f"latent_encoder.{name}"); latent_encoder_param_names_local.append(name)
    buffers_latent_encoder = {name: buffer.to(model_device) for name, buffer in model.latent_encoder.named_buffers()}

    phi_params_tuple_for_jac = tuple(phi_param_tensors) 
    head_idx_for_task = model.task_to_head_map.get(orig_task_id)

    if head_idx_for_task is None or not (0 <= head_idx_for_task < len(model.decoders)):
        print(f"Error OGP: Task {orig_task_id+1} no valid head ({head_idx_for_task}). Skip J."); return torch.empty(0,0,device=model_device) 
    
    decoder_module_for_jac = model.decoders[head_idx_for_task] 

    def compute_output_from_phi_params_static(
            phi_params_rt_tuple, raw_context_x, context_y_onehot, raw_target_x_val, # Raw images
            cnn_mod, xy_enc_mod, latent_enc_mod, agg_fn, dec_mod_fixed, # Modules
            is_img_task, # Boolean
            cnn_pnames_ord, xy_pnames_ord, lat_pnames_ord, # Parameter names for ordering
            cnn_bufs, xy_bufs, lat_bufs, # Buffers
            n_cnn_p, n_xy_p): # Lengths of param lists for indexing phi_params_rt_tuple
        
        # Reconstruct parameter dictionaries for functional_call
        params_cnn_dict = {n: phi_params_rt_tuple[i] for i, n in enumerate(cnn_pnames_ord)} if is_img_task else {}
        params_xy_enc_dict = {n: phi_params_rt_tuple[i + n_cnn_p] for i, n in enumerate(xy_pnames_ord)}
        params_lat_enc_dict = {n: phi_params_rt_tuple[i + n_cnn_p + n_xy_p] for i, n in enumerate(lat_pnames_ord)}

        # --- Feature Extraction ---
        batch_size_ctx_jac, num_ctx_pts_jac, *_ = raw_context_x.shape
        batch_size_trg_jac, num_trg_pts_jac, *_ = raw_target_x_val.shape

        if num_ctx_pts_jac > 0:
            if is_img_task:
                img_c, img_h, img_w = raw_context_x.shape[-3:]
                ctx_features = functional_call(cnn_mod, (params_cnn_dict, cnn_bufs), args=(raw_context_x.reshape(-1, img_c, img_h, img_w),))
                ctx_features = ctx_features.view(batch_size_ctx_jac, num_ctx_pts_jac, -1)
            else: ctx_features = raw_context_x
        else: ctx_features = torch.empty(batch_size_ctx_jac, 0, model.feature_dim_internal, device=model_device) # Use model.feature_dim_internal

        if num_trg_pts_jac > 0:
            if is_img_task:
                img_c, img_h, img_w = raw_target_x_val.shape[-3:]
                trg_features = functional_call(cnn_mod, (params_cnn_dict, cnn_bufs), args=(raw_target_x_val.reshape(-1, img_c, img_h, img_w),))
                trg_features = trg_features.view(batch_size_trg_jac, num_trg_pts_jac, -1)
            else: trg_features = raw_target_x_val
        else: trg_features = torch.empty(batch_size_trg_jac, 0, model.feature_dim_internal, device=model_device)


        # --- NP Path (using features) ---
        r_i = functional_call(xy_enc_mod, (params_xy_enc_dict, xy_bufs), args=(ctx_features, context_y_onehot))
        r_agg = agg_fn(r_i)
        zm, _ = functional_call(latent_enc_mod, (params_lat_enc_dict, lat_bufs), args=(r_agg))
        
        # Decoder (fixed) operates on target features
        yp_flat = dec_mod_fixed(zm, trg_features).flatten() 
        return yp_flat

    model.cnn_feature_extractor.eval(); model.xy_encoder.eval(); model.latent_encoder.eval(); decoder_module_for_jac.eval()

    jacobian_tuple_per_phi_param = jacrev(compute_output_from_phi_params_static, argnums=0, has_aux=False)(
                                        phi_params_tuple_for_jac,
                                        ctx_x_raw_i, ctx_y_oh_i, trg_x_raw_val_i, # Pass raw images
                                        model.cnn_feature_extractor, model.xy_encoder, model.latent_encoder, model.aggregate,
                                        decoder_module_for_jac, model.is_image_task,
                                        cnn_param_names_local, xy_encoder_param_names_local, latent_encoder_param_names_local,
                                        buffers_cnn, buffers_xy_encoder, buffers_latent_encoder,
                                        num_phi_cnn, num_phi_xy_encoder)

    jacobian_matrices_flattened_list = []
    if not jacobian_tuple_per_phi_param or len(jacobian_tuple_per_phi_param) != len(phi_params_tuple_for_jac):
        print(f"CRIT OGP Err: jacrev len mismatch. Exp {len(phi_params_tuple_for_jac)}, got {len(jacobian_tuple_per_phi_param) if jacobian_tuple_per_phi_param else 'None'}.")
        return torch.empty(0,0,device=model_device)

    total_output_dim = trg_x_raw_val_i.size(0) * trg_x_raw_val_i.size(1) * cfg_obj.Y_DIM_OUT 

    for i, jac_for_one_param_tensor in enumerate(jacobian_tuple_per_phi_param):
        param_name_full = phi_param_names_global[i]
        if jac_for_one_param_tensor is None: 
            original_param_numel = phi_params_tuple_for_jac[i].numel()
            print(f"  Warn OGP: jacrev None for phi_param '{param_name_full}'. Zero fill ({total_output_dim},{original_param_numel}).")
            jacobian_matrices_flattened_list.append(torch.zeros(total_output_dim, original_param_numel, device=model_device))
        else:
            jacobian_matrices_flattened_list.append(jac_for_one_param_tensor.reshape(jac_for_one_param_tensor.shape[0], -1))
    
    if not jacobian_matrices_flattened_list: return torch.empty(0,0,device=model_device)
    
    full_jacobian_matrix_task_i = torch.cat(jacobian_matrices_flattened_list, dim=1).detach()
    return full_jacobian_matrix_task_i


# --- project_gradients_ogp, train_ogp_np_task, evaluate_ogp_np_task ---
# These functions should largely remain the same as in ogp_np_experiment_py_v3,
# as they operate on the model interface which now internally handles the CNN.
# The key is that `get_ogp_data_for_jacobian` provides raw images,
# and `collect_jacobian_for_task_ogp_func` correctly uses them.
# And `train_ogp_np_task` and `evaluate_ogp_np_task` also pass raw images to `model.forward`
# and `get_context_target_split`.

# --- Main Experiment Loop (run_continual_learning_ogp_np_experiment) ---
# This function also remains structurally similar.
# The main change is that x_train_current_task and x_test_task will contain raw image tensors (C,H,W)
# instead of flattened vectors for CIFAR/TinyImageNet.
# The calls to model.decide_head_for_task will pass raw context images.

# --- Main Execution Block (__main__) ---
# This also remains structurally similar.

# For brevity, I'll paste the rest of the functions assuming minor adaptations if needed,
# focusing on the parts that interact with data shapes.

# (Paste the rest of the functions: project_gradients_ogp, train_ogp_np_task,
#  evaluate_ogp_np_task, run_continual_learning_ogp_np_experiment, and __main__
#  from ogp_np_experiment_py_v3, ensuring that data passed to model methods
#  like `decide_head_for_task`, `forward`, and `get_ogp_data_for_jacobian`
#  are raw images (e.g., (B, N, C, H, W) or (B, N, D)) and that
#  `get_context_target_split` also handles these raw image shapes correctly.)

# The following functions are largely the same as in ogp_np_experiment_py_v3,
# with minor adjustments for clarity or to ensure raw image data is passed where needed.

def project_gradients_ogp(model,cfg_obj):
    if model.past_task_jacobians_stacked is None or model.past_task_jacobians_stacked.numel()==0:return
    J_old = model.past_task_jacobians_stacked 
    ogp_device = model.device_internal
    phi_params_w_grads = [p for p in model.get_phi_parameters() if p.grad is not None]
    if not phi_params_w_grads: return
    current_grads_flat = torch.cat([p.grad.flatten() for p in phi_params_w_grads]).to(ogp_device)
    if J_old.shape[1] != current_grads_flat.shape[0]:
        print(f"CRIT OGP Dim Err: J_old.col ({J_old.shape[1]})!=grad.dim ({current_grads_flat.shape[0]}). Skip proj."); return
    J_eff = J_old
    if J_old.size(0) > cfg_obj.OPM_PROJ_DIM and cfg_obj.OPM_PROJ_DIM > 0:
        n_con, p_dim_proj = J_old.size(0), min(cfg_obj.OPM_PROJ_DIM, J_old.size(0), J_old.size(1))
        if p_dim_proj > 0:
            try:
                rp_mat = torch.randn(n_con, p_dim_proj, device=ogp_device)
                q_mat, _ = torch.linalg.qr(rp_mat)
                J_eff = q_mat.T @ J_old
            except torch.linalg.LinAlgError as e: J_eff = J_old; print(f"  OGP: QR failed: {e}. Using full J_old.")
        else: J_eff = J_old
    A = J_eff @ J_eff.T 
    A.diagonal().add_(cfg_obj.JACOBIAN_PROJ_REG)
    B = J_eff @ current_grads_flat
    proj_grads_flat = current_grads_flat 
    try:
        L = torch.linalg.cholesky(A)
        x = torch.cholesky_solve(B.unsqueeze(-1),L).squeeze(-1)
        proj_grads_flat = current_grads_flat - J_eff.T @ x
    except torch.linalg.LinAlgError:
        print("  OGP: Cholesky failed. Trying pseudo-inverse.")
        try:
            A_pinv = torch.linalg.pinv(A)
            x = A_pinv @ B
            proj_grads_flat = current_grads_flat - J_eff.T @ x
        except torch.linalg.LinAlgError: print("  OGP: All solves failed. No projection.")
    offset = 0
    for p in phi_params_w_grads:
        numel = p.numel()
        p.grad.data = proj_grads_flat[offset:offset+numel].view_as(p.grad.data)
        offset += numel

def train_ogp_np_task(model,orig_task_idx,assign_head_idx,opt_phi,cfg_obj):
    if not (0<=assign_head_idx<len(model.decoder_optimizers) and assign_head_idx < len(model.decoders)):
        raise ValueError(f"Train Error: Invalid head_idx {assign_head_idx} or optimizers/decoders not properly setup. Decoders: {len(model.decoders)}, Optimizers: {len(model.decoder_optimizers)}")
    optimizer_active_decoder = model.decoder_optimizers[assign_head_idx]
    # x_train_task_raw_images, y_train_task_labels from global list
    x_train_task_raw_images, y_train_task_labels = task_data_train_global[orig_task_idx] 
    if x_train_task_raw_images.numel()==0:
        print(f"Task {orig_task_idx+1}: No training data. Skipping."); return

    train_dataset = TensorDataset(x_train_task_raw_images, y_train_task_labels)
    batch_size_train = min(cfg_obj.BATCH_SIZE_TRAIN, len(train_dataset))
    if batch_size_train == 0:
        print(f"Task {orig_task_idx+1}: Effective batch size 0. Skipping."); return
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, 
                                  drop_last=(len(train_dataset)>batch_size_train and len(train_dataset)%batch_size_train!=0))

    model.train() 
    for i, decoder_module in enumerate(model.decoders):
        is_active_head = (i == assign_head_idx)
        decoder_module.train(mode=is_active_head)
        for param in decoder_module.parameters(): param.requires_grad_(is_active_head)
    for param in model.get_phi_parameters(): param.requires_grad_(True)

    print(f"Train Original Task {orig_task_idx+1} on Head {assign_head_idx} (Epochs: {cfg_obj.EPOCHS_PER_TASK})...")
    for epoch in range(cfg_obj.EPOCHS_PER_TASK):
        epoch_loss_sum, epoch_correct_preds, epoch_total_points, num_batches_processed = 0.0, 0, 0, 0
        for x_batch_raw_images, y_labels_batch in train_dataloader: # x_batch_raw_images are (B, C, H, W) or (B,D)
            x_batch_raw_images = x_batch_raw_images.to(cfg_obj.DEVICE)
            y_labels_batch = y_labels_batch.to(cfg_obj.DEVICE)
            if x_batch_raw_images.size(0)==0: continue 

            # Reshape for NP: (1, B, C, H, W) or (1, B, D)
            # This shape is now handled by get_context_target_split and model.forward
            x_sequence_for_np = x_batch_raw_images.unsqueeze(0) 
            y_sequence_labels_for_np = y_labels_batch.unsqueeze(0)

            x_ctx_raw,y_ctx_onehot,_,x_trg_raw,y_trg_lbl,is_valid_split = get_context_target_split(
                x_sequence_for_np, y_sequence_labels_for_np, model.y_dim_onehot,cfg_obj
            )
            if not is_valid_split: continue 

            opt_phi.zero_grad(set_to_none=True)
            optimizer_active_decoder.zero_grad(set_to_none=True)

            # Model forward takes raw images for context and target
            y_pred, z_mean, z_logvar = model(x_ctx_raw, y_ctx_onehot, x_trg_raw, assign_head_idx) 
            loss, _, _ = ogp_np_loss(y_pred,y_trg_lbl,z_mean,z_logvar,cfg_obj.KL_WEIGHT)

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: NaN/Inf loss epoch {epoch+1}. Skip batch."); continue 
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
            print(f"  Epoch {epoch+1}: No valid batches.")


def evaluate_ogp_np_task(model,orig_task_idx_eval,cfg_obj):
    x_test_task_raw_images, y_test_task_labels = task_data_test_global[orig_task_idx_eval]
    if x_test_task_raw_images.numel()==0: 
        print(f"Eval Task {orig_task_idx_eval+1}: No test data. Acc=0.0"); return 0.0

    head_idx_for_eval = model.task_to_head_map.get(orig_task_idx_eval)
    if head_idx_for_eval is None or not (0<=head_idx_for_eval<len(model.decoders)):
        print(f"Eval Warn: No valid head for task {orig_task_idx_eval+1}. Acc=0."); return 0.0
    
    test_dataset = TensorDataset(x_test_task_raw_images,y_test_task_labels)
    batch_size_eval = min(cfg_obj.BATCH_SIZE_EVAL,len(test_dataset))
    if batch_size_eval == 0: 
        print(f"Eval Task {orig_task_idx_eval+1}: Effective batch size 0. Acc=0.0"); return 0.0
    test_dataloader = DataLoader(test_dataset,batch_size=batch_size_eval,shuffle=False,
                                 drop_last=(len(test_dataset)>batch_size_eval and len(test_dataset)%batch_size_eval!=0))
    
    model.eval(); total_correct_predictions, total_points_evaluated = 0,0
    with torch.no_grad(): 
        for batch_num,(x_batch_raw_images,y_labels_batch) in enumerate(test_dataloader):
            if batch_num >= cfg_obj.NUM_EVAL_BATCHES: break 
            x_batch_raw_images = x_batch_raw_images.to(cfg_obj.DEVICE)
            y_labels_batch = y_labels_batch.to(cfg_obj.DEVICE)
            if x_batch_raw_images.size(0)==0: continue

            num_points_in_batch = x_batch_raw_images.size(0)
            x_sequence_for_np = x_batch_raw_images.unsqueeze(0)
            y_sequence_labels_for_np = y_labels_batch.unsqueeze(0)

            num_ctx_eval = min(cfg_obj.FIXED_EVAL_CONTEXT, num_points_in_batch -1 if num_points_in_batch > 1 else 0)
            num_ctx_eval = max(0, num_ctx_eval) 
            if (num_points_in_batch - num_ctx_eval) <= 0 : continue 

            ctx_indices_eval = torch.arange(num_ctx_eval, device=cfg_obj.DEVICE)
            trg_indices_eval = torch.arange(num_ctx_eval, num_points_in_batch, device=cfg_obj.DEVICE)

            # Select raw images for context and target
            x_ctx_raw_eval, y_ctx_labels_eval = x_sequence_for_np[:,ctx_indices_eval,...], y_sequence_labels_for_np[:,ctx_indices_eval]
            x_trg_raw_eval, y_trg_labels_eval = x_sequence_for_np[:,trg_indices_eval,...], y_sequence_labels_for_np[:,trg_indices_eval]
            
            y_ctx_onehot_eval = F.one_hot(y_ctx_labels_eval.long(),num_classes=model.y_dim_onehot).float() if num_ctx_eval > 0 else \
                                torch.empty(1,0,model.y_dim_onehot, device=cfg_obj.DEVICE,dtype=torch.float)
            if num_ctx_eval == 0: 
                empty_x_shape_eval = list(x_sequence_for_np.shape)
                empty_x_shape_eval[1] = 0 # num_points = 0
                x_ctx_raw_eval = torch.empty(empty_x_shape_eval, device=cfg_obj.DEVICE,dtype=x_sequence_for_np.dtype)
            
            if x_trg_raw_eval.size(1) == 0: continue 

            y_pred_eval,_,_ = model(x_ctx_raw_eval,y_ctx_onehot_eval,x_trg_raw_eval, head_idx_for_eval)
            _,predicted_classes_eval = torch.max(y_pred_eval.data,-1)
            total_correct_predictions += (predicted_classes_eval.squeeze()==y_trg_labels_eval.squeeze()).sum().item()
            total_points_evaluated += y_trg_labels_eval.numel()
    return 100.*total_correct_predictions/total_points_evaluated if total_points_evaluated>0 else 0.0

def run_continual_learning_ogp_np_experiment(cfg_obj):
    model = OGPNP(cfg_obj).to(cfg_obj.DEVICE)
    optimizer_phi = optim.Adam(model.get_phi_parameters(), lr=cfg_obj.LR)
    num_effective_tasks = cfg_obj.NUM_CL_TASKS_EFFECTIVE
    all_tasks_accuracy_matrix = np.full((num_effective_tasks, num_effective_tasks), np.nan)
    avg_accs_stream, active_heads_stream, head_decision_log = [], [], []

    for current_task_idx in range(num_effective_tasks): 
        task_display_number = current_task_idx + 1
        print(f"\n--- Processing Original Task {task_display_number}/{num_effective_tasks} ({cfg_obj.EXPERIMENT_TYPE}) ---")
        model.eval() 
        
        x_train_current_task_raw, y_train_current_task_labels = task_data_train_global[current_task_idx]
        num_decision_context_points = min(x_train_current_task_raw.size(0), cfg_obj.NUM_CONTEXT_JACOBIAN, 100) 
        assigned_head_idx, action_taken = -1, "unknown_initial"

        # Prepare context data (raw images) for head decision
        # This data needs to be (1, N, C, H, W) or (1, N, D)
        if num_decision_context_points > 0 and x_train_current_task_raw.numel() > 0 :
            rand_indices_for_decision = torch.randperm(x_train_current_task_raw.size(0))[:num_decision_context_points]
            ctx_x_raw_for_decision = x_train_current_task_raw[rand_indices_for_decision].to(model.device_internal).unsqueeze(0)
            ctx_y_labels_for_decision = y_train_current_task_labels[rand_indices_for_decision].to(model.device_internal).unsqueeze(0)
            ctx_y_onehot_for_decision = F.one_hot(ctx_y_labels_for_decision.long(),num_classes=model.y_dim_onehot).float()
        else: # Fallback for no/small context data, or if fixed_head_per_task (where Z is not strictly needed for decision)
            print(f"Task {task_display_number}: Using empty/prior context for head decision (num_decision_points: {num_decision_context_points}).")
            empty_x_shape_dec = [1, 0] + list(x_train_current_task_raw.shape[1:]) if x_train_current_task_raw.ndim > 1 else [1,0, cfg_obj.X_DIM if "MNIST" in cfg_obj.EXPERIMENT_TYPE else cfg_obj.CNN_OUTPUT_DIM]
            ctx_x_raw_for_decision = torch.empty(empty_x_shape_dec, device=model.device_internal, 
                                             dtype=x_train_current_task_raw.dtype if x_train_current_task_raw.numel() > 0 else torch.float)
            ctx_y_onehot_for_decision = torch.empty(1,0,model.y_dim_onehot, device=model.device_internal, dtype=torch.float)

        assigned_head_idx, action_taken = model.decide_head_for_task(current_task_idx, ctx_x_raw_for_decision, ctx_y_onehot_for_decision)
        
        head_decision_log.append({
            "task_index": current_task_idx, "task_display_number": task_display_number,
            "assigned_head_index": assigned_head_idx, "action": action_taken,
            "num_active_heads_after_decision": len(model.decoders)
        })
        active_heads_stream.append(len(model.decoders)) 
        print(f"Original Task {task_display_number} assigned to Head {assigned_head_idx} (Action: {action_taken}).")
        
        train_ogp_np_task(model, current_task_idx, assigned_head_idx, optimizer_phi, cfg_obj)
        model.eval()
        
        print(f"Collecting Jacobian for original task {task_display_number} (trained on head {assigned_head_idx})...")
        ctx_x_raw_jac,ctx_y_onehot_jac,val_x_raw_jac = get_ogp_data_for_jacobian(current_task_idx,cfg_obj) # Gets raw images
        if ctx_x_raw_jac is not None and val_x_raw_jac is not None and val_x_raw_jac.size(1)>0: # Check num_points in val_x_raw_jac
            jacobian_current_task = collect_jacobian_for_task_ogp_func(model,current_task_idx,
                                                               ctx_x_raw_jac,ctx_y_onehot_jac,val_x_raw_jac,cfg_obj)
            if jacobian_current_task is not None and jacobian_current_task.numel()>0:
                print(f"  Jacobian for Task {task_display_number} (head {assigned_head_idx}) shape: {jacobian_current_task.shape}")
                total_phi_dimension = sum(p.numel() for p in model.get_phi_parameters())
                if jacobian_current_task.shape[1]!=total_phi_dimension:
                    print(f"  OGP Jacobian Warning: Columns ({jacobian_current_task.shape[1]}) != Total phi params ({total_phi_dimension})")
                else: 
                    if model.past_task_jacobians_stacked is None or model.past_task_jacobians_stacked.numel()==0:
                        model.past_task_jacobians_stacked = jacobian_current_task
                    elif model.past_task_jacobians_stacked.shape[1] != jacobian_current_task.shape[1]: 
                        print("OGP J WARN: Phi dim mismatch! Reset J_stacked."); model.past_task_jacobians_stacked=jacobian_current_task
                    else:
                        model.past_task_jacobians_stacked = torch.cat([model.past_task_jacobians_stacked,jacobian_current_task],dim=0)
                if model.past_task_jacobians_stacked is not None:
                    print(f"  Total J_stacked shape: {model.past_task_jacobians_stacked.shape}")
            else: print(f"  Failed to collect J for task {task_display_number}.")
        else: print(f"  Skipping J collection for task {task_display_number} (no OPM data).")
        
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
    print(f"\nFinal Avg Acc (at end of training): {final_avg_accuracy:.2f}%")
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
        "head_decision_details_per_task": head_decision_details, 
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
    print(header_string); print("-" * len(header_string))
    for eval_task_idx_table in range(num_effective_tasks_table): 
        row_string=f"   Task {eval_task_idx_table+1:<2}    | "
        for training_stage_idx in range(num_effective_tasks_table): 
            accuracy_value = all_tasks_accuracy_matrix[training_stage_idx, eval_task_idx_table]
            if training_stage_idx < eval_task_idx_table: row_string+="   -     | " 
            elif not np.isnan(accuracy_value): row_string+=f"{accuracy_value:7.2f} | "
            else: row_string+="  N/A    | "
        print(row_string)
    if avg_accs_stream: print(f"\nFinal Average Accuracy: {avg_accs_stream[-1]:.2f}%")
    print(f"Final Backward Transfer (BWT): {bwt_final:.4f}")
    print(f"--- Experiment {results_dir_name} Complete ---")

