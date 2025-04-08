import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import argparse
import time
##########################################################
# Other settings
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.randn(1).to(DEVICE) + torch.randn(1).to(DEVICE)

SEED = 42
# --- Configuration ---
FEAT_DIM = 784 # Input dimension
# Set random seed for reproducibility
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# --- Define two classes for Algo-1 and Algo-2 ---
##################################################
# Your code for Task-1 goes here

class EnergyRegressor(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 16),  
            nn.ReLU(inplace=True),
            nn.Linear(16, 8),
            nn.ReLU(inplace=True),
            nn.Linear(8, 4),
            nn.ReLU(inplace=True),
            nn.Linear(4, 2),
            nn.ReLU(inplace=True),
            nn.Linear(2, 1) # Output is a single scalar predicted energy value
        )

    def forward(self, x):
        # Input x should already be flattened [batch, 784]
        return self.net(x)


class Algo1_Sampler:
    def __init__(self, energy_function, epsilon=0.01, n_samples=1000, burn_in=100):
        self.energy_function = energy_function
        self.epsilon = epsilon
        self.n_samples = n_samples
        self.burn_in = burn_in
        
    def sample(self, x_0):
        x_0 = x_0.clone().detach().to(DEVICE)
        x_0.requires_grad_(True)
        
        samples = []
        current_x = x_0.clone()
        
        accepted = 0
        start_time = time.time()
        burn_in_time = None
        for t in range(self.n_samples + self.burn_in):
            current_x.requires_grad_(True)
            energy = self.energy_function(current_x)
            grad_t,  = torch.autograd.grad(energy.sum(), current_x, create_graph=False)
            # current_x.grad.zero_()
            
            omega_t = torch.randn_like(current_x)
            
            proposed_x = current_x - (self.epsilon/2) * grad_t + torch.sqrt(torch.tensor(self.epsilon)) * omega_t
            proposed_x.requires_grad_(True)
            
            proposed_energy = self.energy_function(proposed_x)
            grad_proposed,  = torch.autograd.grad(proposed_energy.sum(), proposed_x, create_graph=False)
            
            log_q_x_given_proposed = -1/(4*self.epsilon) * torch.sum((current_x - (proposed_x - (self.epsilon/2) * grad_proposed))**2)
            log_q_proposed_given_x = -1/(4*self.epsilon) * torch.sum((proposed_x - (current_x - (self.epsilon/2) * grad_t))**2)
            
            acceptance_prob = min(1.0, torch.exp(energy - proposed_energy + log_q_x_given_proposed - log_q_proposed_given_x).item())
            
            u = torch.rand(1).item()
            if u < acceptance_prob:
                current_x = proposed_x.detach().clone()
                current_x.requires_grad_(True)
                accepted += 1
            
            if t >= self.burn_in:
                samples.append(current_x.detach().cpu())
            
            if t == self.burn_in and burn_in_time is None:
                burn_in_time = time.time() - start_time
        # print(accepted)
        acceptance_rate = accepted / self.n_samples
        return samples, acceptance_rate, burn_in_time


class Algo2_Sampler:
    def __init__(self, energy_function, epsilon=0.01, n_samples=1000, burn_in=100):
        self.energy_function = energy_function
        self.epsilon = epsilon
        self.n_samples = n_samples
        self.burn_in = burn_in
        
    def sample(self, x_0):
        x_0 = x_0.clone().detach().to(DEVICE)
        x_0.requires_grad_(True)
        
        samples = []
        current_x = x_0.clone()
        
        start_time = time.time()
        burn_in_time = None
        
        for t in range(self.n_samples + self.burn_in):
            current_x.requires_grad_(True)
            energy = self.energy_function(current_x)
            grad_t,  = torch.autograd.grad(energy.sum(), current_x, create_graph=False)
            
            omega_t = torch.randn_like(current_x)
            
            current_x = current_x.detach() - (self.epsilon/2) * grad_t + torch.sqrt(torch.tensor(self.epsilon)) * omega_t
            # current_x.requires_grad_(True)
            
            if t >= self.burn_in:
                samples.append(current_x.detach().cpu())
            
            if t == self.burn_in and burn_in_time is None:
                burn_in_time = time.time() - start_time
        
        return samples, None, burn_in_time


def visualize_samples(samples_algo1, samples_algo2, title="MCMC Samples", filename_prefix="mcmc_samples"):
    samples_algo1_flat = torch.stack(samples_algo1).view(len(samples_algo1), -1)
    samples_algo2_flat = torch.stack(samples_algo2).view(len(samples_algo2), -1)
    
    all_samples = torch.cat([samples_algo1_flat, samples_algo2_flat], dim=0)
    # print(all_samples)
    labels = np.concatenate([np.zeros(len(samples_algo1)), np.ones(len(samples_algo2))])

    tsne = TSNE(n_components=2, random_state=SEED)
    samples_tsne = tsne.fit_transform(all_samples.numpy())
    # print(samples_tsne)

    plt.figure(figsize=(12, 10))
    
    plt.scatter(
        samples_tsne[labels == 0, 0],
        samples_tsne[labels == 0, 1],
        alpha=0.5,
        label="Algorithm 1 (MH-MCMC)"
    )

    plt.scatter(
        samples_tsne[labels == 1, 0],
        samples_tsne[labels == 1, 1],
        alpha=0.5,
        label="Algorithm 2 (Langevin)"
    )
    
    plt.title(title)
    plt.legend()
    plt.savefig(f"{filename_prefix}_tsne_2d.png")
    plt.close()
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    tsne_3d = TSNE(n_components=3, random_state=SEED)
    samples_tsne_3d = tsne_3d.fit_transform(all_samples.numpy())
    
    ax.scatter(
        samples_tsne_3d[labels == 0, 0],
        samples_tsne_3d[labels == 0, 1],
        samples_tsne_3d[labels == 0, 2],
        alpha=0.5,
        label="Algorithm 1 (MH-MCMC)"
    )
    
    ax.scatter(
        samples_tsne_3d[labels == 1, 0],
        samples_tsne_3d[labels == 1, 1],
        samples_tsne_3d[labels == 1, 2],
        alpha=0.5,
        label="Algorithm 2 (Langevin)"
    )
    
    ax.set_title(title + " (3D)")
    ax.legend()
    plt.savefig(f"{filename_prefix}_tsne_3d.png")
    plt.close()

if __name__ == "__main__":    
    model = EnergyRegressor(FEAT_DIM).to(DEVICE)
    model_weights_path = '../trained_model_weights.pth'
    model.load_state_dict(torch.load(model_weights_path, map_location=DEVICE))
    model.eval()
    parser = argparse.ArgumentParser(description="Run MCMC sampling with two algorithms.")
    parser.add_argument("--n_samples", type=int, default=1000, help="Number of samples to generate")
    parser.add_argument("--burn_in", type=int, default=500, help="Number of burn-in steps")
    parser.add_argument("--epsilon", type=float, default=0.9, help="Step size for Langevin dynamics")

    args = parser.parse_args()

    n_samples = args.n_samples
    burn_in = args.burn_in
    epsilon = args.epsilon
    x_0 = torch.randn(1, FEAT_DIM).to(DEVICE)
    
    print("\nRunning Algorithm 1 (MH-MCMC)...")
    algo1_sampler = Algo1_Sampler(model, epsilon=epsilon, n_samples=n_samples, burn_in=burn_in)
    samples_algo1, acceptance_rate_algo1, burn_in_time_algo1 = algo1_sampler.sample(x_0)
    # print(samples_algo1)
    print(f"Algorithm 1 - Acceptance Rate: {acceptance_rate_algo1:.4f}")
    print(f"Algorithm 1 - Time to Burn-in: {burn_in_time_algo1:.4f} seconds")
    print(f"Algorithm 1 - Generated {len(samples_algo1)} samples after burn-in")
    sum = 0
    for sample in samples_algo1:
        sample = sample.to(DEVICE)
        sum += torch.exp(-model(sample))
    sum /= len(samples_algo1)
    print("Mean probability for Algo1: ", sum)    
    print("\nRunning Algorithm 2 (Langevin)...")
    algo2_sampler = Algo2_Sampler(model, epsilon=epsilon, n_samples=n_samples, burn_in=burn_in)
    samples_algo2, _, burn_in_time_algo2 = algo2_sampler.sample(x_0)
    # print(samples_algo2)
    print(f"Algorithm 2 - Time to Burn-in: {burn_in_time_algo2:.4f} seconds")
    print(f"Algorithm 2 - Generated {len(samples_algo2)} samples after burn-in")

    sum = 0
    for sample in samples_algo2:
        sample = sample.to(DEVICE)
        sum += torch.exp(-model(sample))
    sum /= len(samples_algo1)
    print("Mean probability for Algo2: ", sum)  
    
    filename_prefix = f"samples_eps{epsilon}_n{n_samples}_burn{burn_in}"
    visualize_samples(samples_algo1, samples_algo2, title=f"MCMC Samples (ε={epsilon})", filename_prefix=filename_prefix)
