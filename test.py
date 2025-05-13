import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

# Configuration
class Config:
    SEED = 42
    BATCH_SIZE = 256
    Z_DIM = 32
    IMG_SIZE = 28*28
    SAMPLE_SIZE = 512  # Number of samples per task for distribution estimation
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = Config()

# 1. Define encoder architecture (simplified version from original NP model)
class TaskEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(config.IMG_SIZE, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
        )
        self.fc_mean = torch.nn.Linear(128, config.Z_DIM)
        self.fc_logvar = torch.nn.Linear(128, config.Z_DIM)
        
    def forward(self, x):
        h = self.encoder(x)
        return self.fc_mean(h), self.fc_logvar(h)

# 2. KL divergence functions
def symmetric_kl_divergence(mu1, logvar1, mu2, logvar2):
    """Compute symmetric KL between two diagonal Gaussians"""
    var1, var2 = torch.exp(logvar1), torch.exp(logvar2)
    kl_1_2 = 0.5 * (torch.sum(var1/var2 + (mu2 - mu1)**2/var2 + logvar2 - logvar1 - 1, dim=1))
    kl_2_1 = 0.5 * (torch.sum(var2/var1 + (mu1 - mu2)**2/var1 + logvar1 - logvar2 - 1, dim=1))
    return torch.mean(kl_1_2 + kl_2_1).item()

# 3. Load MNIST and create task sequence
def get_task_data(task_classes):
    """Get data for specific digit classes"""
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    full_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    
    # Create mask for selected classes
    targets = full_dataset.targets.numpy()
    mask = np.isin(targets, task_classes)
    indices = np.where(mask)[0]
    
    return Subset(full_dataset, indices)

# 4. Compute task distributions
def compute_task_distribution(task_data, encoder):
    """Compute latent distribution parameters for a task"""
    loader = DataLoader(task_data, batch_size=config.BATCH_SIZE, shuffle=True)
    
    mus, logvars = [], []
    with torch.no_grad():
        for batch, (x, _) in enumerate(loader):
            x = x.view(-1, config.IMG_SIZE).to(config.DEVICE)
            mu, logvar = encoder(x)
            mus.append(mu.cpu())
            logvars.append(logvar.cpu())
            if len(mus)*x.shape[0] > config.SAMPLE_SIZE:
                break  # Use first few batches for estimation
    
    return torch.cat(mus).mean(dim=0), torch.cat(logvars).mean(dim=0)

# 5. Main execution
def main():
    torch.manual_seed(config.SEED)
    
    # Define our custom task sequence
    task_sequence = [(0,1), (2,3), (4,5), (0,1), (8,9), (8,9), (0,1), (6,7)]  # Notice duplicate (0,1)
    
    # Initialize encoder
    encoder = TaskEncoder().to(config.DEVICE)
    encoder.eval()
    
    # Compute distributions for all tasks
    task_distributions = []
    for task_idx, classes in enumerate(task_sequence):
        print(f"Processing task {task_idx+1}: classes {classes}")
        task_data = get_task_data(classes)
        mu, logvar = compute_task_distribution(task_data, encoder)
        task_distributions.append((mu, logvar))
    
    # Compute pairwise symmetric KL divergences
    n_tasks = len(task_sequence)
    kl_matrix = np.zeros((n_tasks, n_tasks))
    
    for i in range(n_tasks):
        for j in range(n_tasks):
            mu_i, logvar_i = task_distributions[i]
            mu_j, logvar_j = task_distributions[j]
            kl_matrix[i,j] = symmetric_kl_divergence(
                mu_i.unsqueeze(0), logvar_i.unsqueeze(0),
                mu_j.unsqueeze(0), logvar_j.unsqueeze(0)
            )

    # Visualization
    plt.figure(figsize=(10,8))
    plt.imshow(kl_matrix, cmap='viridis', origin='upper')
    plt.colorbar(label='Symmetric KL Divergence')
    plt.xticks(range(n_tasks), [f"Task {i+1}\n{task_sequence[i]}" for i in range(n_tasks)])
    plt.yticks(range(n_tasks), [f"Task {i+1}\n{task_sequence[i]}" for i in range(n_tasks)])
    plt.title("Pairwise Symmetric KL Divergence Between Task Distributions")
    plt.tight_layout()
    # plt.show()
    plt.savefig("kl_divergence_matrix.png")
    plt.close()

if __name__ == "__main__":
    main()