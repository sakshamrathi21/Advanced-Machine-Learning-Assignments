import os
import time

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# --- Configuration ---
FEAT_DIM = 784 # Input dimension

# Other settings
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

# Set random seed for reproducibility
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

print(f"Using device: {DEVICE}")

# Custom Dataset for loading the synthetic (x, E(x)) data
class EnergyDataset(Dataset):
    def __init__(self, filepath):
        print(f"Loading dataset from {filepath}...")
        start_time = time.time()
        try:
            data_dict = torch.load(filepath)
            self.x = data_dict['x']
            self.energy = data_dict['energy'].unsqueeze(1) # Ensure energy has shape [N, 1] for regression
            load_time = time.time() - start_time
            print(f"Dataset loaded in {load_time:.2f}s. Shape: x={self.x.shape}, energy={self.energy.shape}")
        except FileNotFoundError:
            print(f"Error: Dataset file not found at {filepath}")
            raise
        except Exception as e:
            print(f"Error loading dataset: {e}")
            raise

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.energy[idx]

# Neural Network Model E_theta for Energy Regression
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

# Trainer Class
class Tester:
    def __init__(self, model, val_loader, criterion, device):
        self.model = model.to(device)
        self.val_loader = val_loader
        self.criterion = criterion # Loss function (e.g., MSELoss)
        self.device = device
        
    def test(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for x_batch, energy_batch in self.val_loader:
                x_batch, energy_batch = x_batch.to(self.device), energy_batch.to(self.device)
                predicted_energy = self.model(x_batch)
                loss = self.criterion(predicted_energy, energy_batch)
                total_loss += loss.item() * x_batch.size(0)

        avg_loss = total_loss / len(self.val_loader.dataset)
        return avg_loss

# --- Main Execution Logic for Training ---
if __name__ == "__main__":


    #################################################################
    # DO NOT MAKE ANY CHANGES ABOVE THIS LINE
    # Write your code for TASK 0 below
    # TASK 0: Initialize Model and load weights
    DATASET_PATH = '/PATH/TO/TEST/DATASET'  # Path to the dataset file
    
    model = 
    # Load the model weights
    
    
    
    
    
    ##################################################################
    # DO NOT CHANGE ANYTHING BELOW THIS LINE
    # MSE LOSS
    criterion = nn.MSELoss() # Mean Squared Error for regression

    print("\n--- Model Architecture ---")
    print(model)
    print("------------------------\n")

    # Test dataset
    # Load the test dataset
    try:
        test_dataset = EnergyDataset(DATASET_PATH)
    except Exception:
        print("Failed to load dataset. Exiting.")
        exit()
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False, num_workers=2, pin_memory=True)
    test = Tester(model, test_loader, criterion, DEVICE)
    test_loss = test.test()
    
    print("\n--- Test Results ---")
    print(f"Loss: {test_loss:.4f}")
    # Report this loss as a result of TASK 0.
    print("--- Script Finished ---")