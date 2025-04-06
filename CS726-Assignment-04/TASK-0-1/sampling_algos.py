import os
import torch
import numpy as np
import matplotlib.pyplot as plt
# You can import any other torch modules you need below #



##########################################################

# Other settings
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

# Set random seed for reproducibility
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# --- Define two classes for Algo-1 and Algo-2 ---
##################################################
# Your code for Task-1 goes here
class Algo1_Sampler:
    
class Algo2_Sampler:
    
# --- Main Execution ---
if __name__ == "__main__":
    
    