import torch
import torch.utils
import torch.utils.data
from tqdm.auto import tqdm
from torch import nn
import argparse
import torch.nn.functional as F
import utils
import dataset
import os
import matplotlib.pyplot as plt
import math
from utils import get_nll, get_emd

class SinusoidalPositionEmbeddings(nn.Module):
    """
    Sinusoidal time embedding for the DDPM model
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        """
        Args:
            time: torch.Tensor, the timestep tensor [batch_size]
            
        Returns:
            torch.Tensor, the time embedding [batch_size, dim]
        """
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        
        # If dimension is odd, pad with zeros
        if self.dim % 2 == 1:
            embeddings = F.pad(embeddings, (0, 1, 0, 0))
            
        return embeddings

class ResidualBlock(nn.Module):
    """
    Residual block for U-Net-like architecture adapted for vector data
    """
    def __init__(self, in_channels, out_channels, time_channels, use_attention=False):
        super().__init__()
        self.use_attention = use_attention
        
        # First linear layer
        self.norm1 = nn.LayerNorm(in_channels)
        self.act1 = nn.SiLU()
        self.linear1 = nn.Linear(in_channels, out_channels)
        
        # Time projection
        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_channels, out_channels)
        )
         
        # Second linear layer
        self.norm2 = nn.LayerNorm(out_channels)
        self.act2 = nn.SiLU()
        self.linear2 = nn.Linear(out_channels, out_channels)
        
        # Residual connection if input and output dimensions differ
        self.residual_connection = nn.Linear(in_channels, out_channels) if in_channels != out_channels else nn.Identity()
        
        # Optional attention layer
        if use_attention:
            self.attention = MultiHeadAttention(out_channels, num_heads=4)
            self.attention_norm = nn.LayerNorm(out_channels)
    
    def forward(self, x, time_emb):
        """
        Args:
            x: torch.Tensor, input feature tensor [batch_size, in_channels]
            time_emb: torch.Tensor, time embedding [batch_size, time_channels]
            
        Returns:
            torch.Tensor, output feature tensor [batch_size, out_channels]
        """
        # Residual path
        residual = self.residual_connection(x)
        
        # Main path
        h = self.norm1(x)
        h = self.act1(h)
        h = self.linear1(h)
        
        # Add time embedding
        time_projection = self.time_proj(time_emb)
        h = h + time_projection
        
        # Second part of main path
        h = self.norm2(h)
        h = self.act2(h)
        h = self.linear2(h)
        
        # Apply attention if specified
        if self.use_attention:
            h = h + self.attention(self.attention_norm(h))
        
        # Add residual connection
        return h + residual

class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention module for vector data
    """
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        assert self.head_dim * num_heads == channels, "channels must be divisible by num_heads"
        
        self.query = nn.Linear(channels, channels)
        self.key = nn.Linear(channels, channels)
        self.value = nn.Linear(channels, channels)
        self.out_proj = nn.Linear(channels, channels)
        
    def forward(self, x):
        """
        Args:
            x: torch.Tensor, input feature tensor [batch_size, channels]
            
        Returns:
            torch.Tensor, attention output [batch_size, channels]
        """
        batch_size = x.shape[0]
        
        # Reshape for multi-head attention
        q = self.query(x).view(batch_size, self.num_heads, self.head_dim)
        k = self.key(x).view(batch_size, self.num_heads, self.head_dim)
        v = self.value(x).view(batch_size, self.num_heads, self.head_dim)
        
        # Compute attention
        # attention = torch.einsum('bhd,bhd->bh', q, k) / (self.head_dim ** 0.5)
        attention = torch.einsum('bhd,bhd->bh', q, k)
        attention = F.softmax(attention, dim=-1)
        
        # Apply attention to values
        out = torch.einsum('bh,bhd->bhd', attention, v)
        out = out.reshape(batch_size, -1)
        return self.out_proj(out)

class UNetVectorModel(nn.Module):
    """
    U-Net-like architecture adapted for vector data
    """
    def __init__(self, n_dim, time_emb_dim=128, model_channels=128, channel_mults=(1, 2, 4, 8)):
        super().__init__()
        self.time_emb_dim = time_emb_dim
        
        # Time embedding
        self.time_embedding = SinusoidalPositionEmbeddings(time_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim)
        )
        
        # Input projection
        self.input_proj = nn.Linear(n_dim, model_channels)
        
        # Down blocks
        self.down_blocks = nn.ModuleList()
        current_channels = model_channels
        down_channel_list = []
        
        for mult in channel_mults:
            out_channels = model_channels * mult
            
            # Add residual blocks with increasing channel dimensions
            self.down_blocks.append(
                ResidualBlock(current_channels, out_channels, time_emb_dim, use_attention=(mult >= 4))
            )
            
            down_channel_list.append(current_channels)
            current_channels = out_channels
        
        # Middle block with attention
        self.middle_block1 = ResidualBlock(current_channels, current_channels, time_emb_dim, use_attention=True)
        self.middle_block2 = ResidualBlock(current_channels, current_channels, time_emb_dim, use_attention=False)
        
        # Up blocks
        self.up_blocks = nn.ModuleList()
        
        for mult in reversed(channel_mults):
            out_channels = model_channels * mult
            
            # Skip connection from down blocks
            skip_channels = down_channel_list.pop()
            
            # Add residual blocks with decreasing channel dimensions
            self.up_blocks.append(
                ResidualBlock(current_channels + skip_channels, out_channels, time_emb_dim, use_attention=(mult >= 4))
            )
            
            current_channels = out_channels
        
        # Output projection
        self.norm_out = nn.LayerNorm(current_channels)
        self.act_out = nn.SiLU()
        self.out_proj = nn.Linear(current_channels, n_dim)
        
    def forward(self, x, t):
        """
        Args:
            x: torch.Tensor, input tensor [batch_size, n_dim]
            t: torch.Tensor, timestep tensor [batch_size]
            
        Returns:
            torch.Tensor, predicted noise [batch_size, n_dim]
        """
        # Time embedding
        time_emb = self.time_embedding(t)
        time_emb = self.time_mlp(time_emb)
        
        # Initial projection
        h = self.input_proj(x)
        
        # Store skip connections
        skips = []
        
        # Down path
        for down_block in self.down_blocks:
            skips.append(h)
            h = down_block(h, time_emb)
        
        # Middle
        h = self.middle_block1(h, time_emb)
        h = self.middle_block2(h, time_emb)
        
        # Up path with skip connections
        for up_block in self.up_blocks:
            # Add skip connection
            skip = skips.pop()
            h = torch.cat([h, skip], dim=-1)
            h = up_block(h, time_emb)
        
        # Output projection
        h = self.norm_out(h)
        h = self.act_out(h)
        h = self.out_proj(h)
        
        return h
    
class NoiseScheduler():
    """
    Noise scheduler for the DDPM model

    Args:
        num_timesteps: int, the number of timesteps
        type: str, the type of scheduler to use
        **kwargs: additional arguments for the scheduler

    This object sets up all the constants like alpha, beta, sigma, etc. required for the DDPM model
    
    """
    def __init__(self, num_timesteps=50, type="linear", **kwargs):

        self.num_timesteps = num_timesteps
        self.type = type

        if type == "linear":
            self.init_linear_schedule(**kwargs)
        else:
            raise NotImplementedError(f"{type} scheduler is not implemented") # change this if you implement additional schedulers


    def init_linear_schedule(self, beta_start, beta_end):
        """
        Precompute whatever quantities are required for training and sampling
        """

        self.betas = torch.linspace(beta_start, beta_end, self.num_timesteps, dtype=torch.float32)
        self.alphas = 1.0 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, 0)

    def __len__(self):
        return self.num_timesteps
    
class DDPM(nn.Module):
    def __init__(self, n_dim=3, n_steps=200):
        """
        Noise prediction network for the DDPM

        Args:
            n_dim: int, the dimensionality of the data
            n_steps: int, the number of steps in the diffusion process
        """
        super().__init__()
        self.n_dim = n_dim
        self.n_steps = n_steps
        self.model = UNetVectorModel(n_dim)

    def forward(self, x, t):
        """
        Args:
            x: torch.Tensor, the input data tensor [batch_size, n_dim]
            t: torch.Tensor, the timestep tensor [batch_size]

        Returns:
            torch.Tensor, the predicted noise tensor [batch_size, n_dim]
        """
        return self.model(x, t)

class ConditionalDDPM(nn.Module):
    def __init__(self, n_dim = 3, n_steps = 200, n_classes = 10):
        super().__init__()
        self.n_dim = n_dim
        self.n_steps = n_steps
        self.n_classes = n_classes
        self.model = UNetVectorModel(n_dim + n_classes)
    
    def forward(self, x, t, class_labels = None):
        batch_size = x.shape[0]
        device = x.device
        if class_labels is None:
            class_encoding = torch.zeros(batch_size, self.n_classes, device=device)
        else:
            class_encoding = F.one_hot(class_labels, num_classes=self.n_classes).float().to(device)
        conditioned_x = torch.cat([x, class_encoding], dim=1)
        noise_pred = self.model(conditioned_x, t)
        return noise_pred[:, :self.n_dim]

    
class ClassifierDDPM():
    """
    ClassifierDDPM implements a classification algorithm using the DDPM model
    """
    
    def __init__(self, model: ConditionalDDPM, noise_scheduler: NoiseScheduler):
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.device = next(model.parameters()).device

    def __call__(self, x):
        return self.predict(x)

    def predict(self, x):
        probs = self.predict_proba(x)
        return torch.argmax(probs, dim=1)

    def predict_proba(self, x):
        batch_size = x.shape[0]
        n_classes = self.model.n_classes
        device = self.device
        x = x.to(device)
        t = torch.ones(batch_size, dtype=torch.long, device=device) * (self.noise_scheduler.num_timesteps // 2)
        alpha_bar_t = self.noise_scheduler.alpha_bar[t].view(-1, 1)
        noise = torch.randn_like(x, device=device)
        x_t = torch.sqrt(alpha_bar_t) * x + torch.sqrt(1 - alpha_bar_t) * noise
        class_scores = torch.zeros((batch_size, n_classes), device=device)
        for c in range(n_classes):
            class_labels = torch.full((batch_size,), c, dtype=torch.long, device=device)
            predicted_noise = self.model(x_t, t, class_labels)
            error = torch.mean((predicted_noise - noise)**2, dim=1)
            class_scores[:, c] = -error
        probs = F.softmax(class_scores, dim=1)
        return probs
    
def trainConditional(model, noise_scheduler, dataloader, optimizer, epochs, run_name):
    """
    Train the model and save the model and necessary plots

    Args:
        model: DDPM, model to train
        noise_scheduler: NoiseScheduler, scheduler for the noise
        dataloader: torch.utils.data.DataLoader, dataloader for the dataset
        optimizer: torch.optim.Optimizer, optimizer to use
        epochs: int, number of epochs to train the model
        run_name: str, path to save the model
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train()
    loss_fn = nn.MSELoss()
    loss_history = []
    for epoch in range(epochs):
        epoch_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for x, y in progress_bar:
            x = x.to(device)
            t = torch.randint(0, noise_scheduler.num_timesteps, (x.shape[0],), device=device)
            noise = torch.randn_like(x, device=device)
            alpha_bar_t = noise_scheduler.alpha_bar[t].view(-1, 1)
            alpha_bar_t = alpha_bar_t.to(device)
            x_t = torch.sqrt(alpha_bar_t) * x + torch.sqrt(1 - alpha_bar_t) * noise
            pred = model(x_t, t, y)
            loss = loss_fn(pred, noise) 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
        avg_epoch_loss = epoch_loss / len(dataloader)
        loss_history.append(avg_epoch_loss)
        print(f"Epoch {epoch+1}/{epochs} Loss: {avg_epoch_loss}")

    os.makedirs(run_name, exist_ok=True)
    model_path = os.path.join(run_name, "model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # Plot loss curve
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), loss_history, marker="o", linestyle="-")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.grid(True)
    plt.savefig(os.path.join(run_name, "loss_curve.png"))

def train(model, noise_scheduler, dataloader, optimizer, epochs, run_name):
    """
    Train the model and save the model and necessary plots

    Args:
        model: DDPM, model to train
        noise_scheduler: NoiseScheduler, scheduler for the noise
        dataloader: torch.utils.data.DataLoader, dataloader for the dataset
        optimizer: torch.optim.Optimizer, optimizer to use
        epochs: int, number of epochs to train the model
        run_name: str, path to save the model
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train()
    loss_fn = nn.MSELoss()
    loss_history = []
    for epoch in range(epochs):
        epoch_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for x in progress_bar:
            x = x[0].to(device)
            t = torch.randint(0, noise_scheduler.num_timesteps, (x.shape[0],), device=device)
            noise = torch.randn_like(x, device=device)
            alpha_bar_t = noise_scheduler.alpha_bar[t].view(-1, 1)
            alpha_bar_t = alpha_bar_t.to(device)
            x_t = torch.sqrt(alpha_bar_t) * x + torch.sqrt(1 - alpha_bar_t) * noise
            pred = model(x_t, t)
            loss = loss_fn(pred, noise) 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
        avg_epoch_loss = epoch_loss / len(dataloader)
        loss_history.append(avg_epoch_loss)
        print(f"Epoch {epoch+1}/{epochs} Loss: {avg_epoch_loss}")

    os.makedirs(run_name, exist_ok=True)
    model_path = os.path.join(run_name, "model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # Plot loss curve
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), loss_history, marker="o", linestyle="-")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.grid(True)
    
    plt.savefig(os.path.join(run_name, "loss_curve.png"))


@torch.no_grad()
def sample(model, n_samples, noise_scheduler, return_intermediate=False): 
    """
    Sample from the model
    
    Args:
        model: DDPM
        n_samples: int
        noise_scheduler: NoiseScheduler
        return_intermediate: bool
    Returns:
        torch.Tensor, samples from the model [n_samples, n_dim]

    If `return_intermediate` is `False`,
            torch.Tensor, samples from the model [n_samples, n_dim]
    Else
        the function returns all the intermediate steps in the diffusion process as well 
        Return: [[n_samples, n_dim]] x n_steps
        Optionally implement return_intermediate=True, will aid in visualizing the intermediate steps
    """ 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_dim = model.n_dim
    x_t = torch.randn((n_samples, n_dim), device=device)  # Start from Gaussian noise
    intermediate_steps = [] if return_intermediate else None
    
    for t in reversed(range(noise_scheduler.num_timesteps)):
        if t == 0:
            break
        t_tensor = torch.full((n_samples,), t, device=device, dtype=torch.long)
        noise_pred = model(x_t, t_tensor)
        alpha_bar_t = noise_scheduler.alpha_bar[t]
        alpha_t = noise_scheduler.alphas[t]
        beta_t = noise_scheduler.betas[t]
        mean = (1 / torch.sqrt(alpha_t)) * (x_t - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * noise_pred)
        if t > 1:
            noise = torch.randn_like(x_t, device=device)
            std = (1 - noise_scheduler.alpha_bar[t - 1]) / (1 - alpha_bar_t) * beta_t
            x_t = mean + torch.sqrt(std) * noise
        else:
            x_t = mean
        
        if return_intermediate:
            intermediate_steps.append(x_t.clone().detach())

    x_t_np = x_t.cpu().numpy()
    plt.figure(figsize=(6, 6))
    plt.scatter(x_t_np[:, 0], x_t_np[:, 1], alpha=0.6, s=10)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Final Sampled Points")
    plt.grid()
    plt.axis('equal')
    plt.savefig("moon.png")
    return (x_t if not return_intermediate else intermediate_steps)

@torch.no_grad()
def sampleConditional(model, n_samples, noise_scheduler, class_label=None, return_intermediate=False, save_plot=True): 
    """
    Sample from the conditional model
    
    Args:
        model: ConditionalDDPM
        n_samples: int
        noise_scheduler: NoiseScheduler
        class_label: int or None, the class label to condition on. If None, samples are generated unconditionally.
        return_intermediate: bool, whether to return intermediate steps
        save_plot: bool, whether to save the plot
        
    Returns:
        If `return_intermediate` is `False`:
            torch.Tensor, samples from the model [n_samples, n_dim]
        Else:
            list of torch.Tensor, intermediate steps of the diffusion process
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_dim = model.n_dim
    x_t = torch.randn((n_samples, n_dim), device=device)  # Start from Gaussian noise
    if class_label is not None:
        class_labels = torch.full((n_samples,), class_label, dtype=torch.long, device=device)
    else:
        class_labels = None
    intermediate_steps = [] if return_intermediate else None
    
    for t in reversed(range(noise_scheduler.num_timesteps)):
        if t == 0:
            break
            
        t_tensor = torch.full((n_samples,), t, device=device, dtype=torch.long)
        noise_pred = model(x_t, t_tensor, class_labels)
        alpha_bar_t = noise_scheduler.alpha_bar[t]
        alpha_t = noise_scheduler.alphas[t]
        beta_t = noise_scheduler.betas[t]
        mean = (1 / torch.sqrt(alpha_t)) * (x_t - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * noise_pred)
        if t > 1:
            noise = torch.randn_like(x_t, device=device)
            std = (1 - noise_scheduler.alpha_bar[t - 1]) / (1 - alpha_bar_t) * beta_t
            x_t = mean + torch.sqrt(std) * noise
        else:
            x_t = mean
            
        if return_intermediate:
            intermediate_steps.append(x_t.clone().detach())
    if save_plot and class_label is not None:
        x_t_np = x_t.cpu().numpy()
        plt.figure(figsize=(6, 6))
        plt.scatter(x_t_np[:, 0], x_t_np[:, 1], alpha=0.6, s=10)
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.title(f"Sampled Points for Class {class_label}")
        plt.grid()
        plt.axis('equal')
        plt.savefig(f"conditional_samples_class_{class_label}.png")
        
    return (x_t if not return_intermediate else intermediate_steps)


def sampleMultipleClasses(model, n_samples_per_class, noise_scheduler, n_classes, run_name):
    """
    Sample multiple classes from the conditional model and plot them with different colors
    
    Args:
        model: ConditionalDDPM
        n_samples_per_class: int, number of samples per class
        noise_scheduler: NoiseScheduler
        n_classes: int, number of classes to sample
        run_name: str, path to save the plot
    
    Returns:
        dict of torch.Tensor, samples for each class {class_label: samples}
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_samples = {}
    colors = plt.cm.get_cmap('tab10', n_classes)
    
    plt.figure(figsize=(10, 8))
    
    for class_label in range(n_classes):
        samples = sampleConditional(
            model, 
            n_samples_per_class, 
            noise_scheduler, 
            class_label=class_label, 
            save_plot=False
        )
    
        all_samples[class_label] = samples
        samples_np = samples.cpu().numpy()
        plt.scatter(
            samples_np[:, 0], 
            samples_np[:, 1], 
            alpha=0.7, 
            s=15, 
            color=colors(class_label),
            label=f"Class {class_label}"
        )
    
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title(f"Samples from Conditional DDPM for All Classes")
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.savefig(f"{run_name}/multi_class_samples.png")
    plt.close()
    
    return all_samples


def sampleCFG(model, n_samples, noise_scheduler, guidance_scale, class_label):
    """
    Sample from the conditional model
    
    Args:
        model: ConditionalDDPM
        n_samples: int
        noise_scheduler: NoiseScheduler
        guidance_scale: float
        class_label: int

    Returns:
        torch.Tensor, samples from the model [n_samples, n_dim]
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_dim = model.n_dim
    x_t = torch.randn((n_samples, n_dim), device=device)
    class_labels = torch.full((n_samples,), class_label, dtype=torch.long, device=device)
    for t in reversed(range(noise_scheduler.num_timesteps)):
        if t == 0:
            break
        t_tensor = torch.full((n_samples,), t, device=device, dtype=torch.long)
        noise_pred_conditional = model(x_t, t_tensor, class_labels)
        noise_pred_unconditional = model(x_t, t_tensor, None)
        noise_pred = noise_pred_unconditional + guidance_scale * (noise_pred_conditional - noise_pred_unconditional)
        alpha_bar_t = noise_scheduler.alpha_bar[t]
        alpha_t = noise_scheduler.alphas[t]
        beta_t = noise_scheduler.betas[t]
        mean = (1 / torch.sqrt(alpha_t)) * (x_t - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * noise_pred)
        if t > 1:
            noise = torch.randn_like(x_t, device=device)
            std = (1 - noise_scheduler.alpha_bar[t - 1]) / (1 - alpha_bar_t) * beta_t
            x_t = mean + torch.sqrt(std) * noise
        else:
            x_t = mean
    return x_t

def sampleSVDD(model, n_samples, noise_scheduler, reward_scale, reward_fn):
    """
    Sample from the SVDD model

    Args:
        model: ConditionalDDPM
        n_samples: int
        noise_scheduler: NoiseScheduler
        reward_scale: float
        reward_fn: callable, takes in a batch of samples torch.Tensor:[n_samples, n_dim] and returns torch.Tensor[n_samples]

    Returns:
        torch.Tensor, samples from the model [n_samples, n_dim]
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_dim = model.n_dim

    x_t = torch.randn((n_samples, n_dim), device=device)

    for t in reversed(range(noise_scheduler.num_timesteps)):
        if t == 0:
            break
        
        t_tensor = torch.full((n_samples,), t, device=device, dtype=torch.long)
    
        noise_pred = model(x_t, t_tensor)
        
        alpha_bar_t = noise_scheduler.alpha_bar[t]
        alpha_t = noise_scheduler.alphas[t]
        beta_t = noise_scheduler.betas[t]
        x_0_pred = (x_t - torch.sqrt(1 - alpha_bar_t) * noise_pred) / torch.sqrt(alpha_bar_t)

        if reward_scale > 0:
            rewards = reward_fn(x_0_pred)
            eps = 1e-4
            grad_rewards = torch.zeros_like(x_t)
            
            for i in range(n_dim):
                x_0_perturbed = x_0_pred.clone()
                x_0_perturbed[:, i] += eps
                rewards_perturbed = reward_fn(x_0_perturbed)
                grad_rewards[:, i] = (rewards_perturbed - rewards) / eps
            grad_rewards = grad_rewards * reward_scale
            noise_pred = noise_pred - torch.sqrt(1 - alpha_bar_t) * grad_rewards
        mean = (1 / torch.sqrt(alpha_t)) * (x_t - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * noise_pred)
        if t > 1:
            noise = torch.randn_like(x_t, device=device)
            std = (1 - noise_scheduler.alpha_bar[t - 1]) / (1 - alpha_bar_t) * beta_t
            x_t = mean + torch.sqrt(std) * noise
        else:
            x_t = mean
    
    return x_t
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=['train', 'sample', 'train_conditional', 'sample_conditional', 'sample_cfg', 'sample_multi_class'], default='sample')
    parser.add_argument("--n_steps", type=int, default=None)
    parser.add_argument("--lbeta", type=float, default=None)
    parser.add_argument("--ubeta", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--n_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_dim", type=int, default=None)
    parser.add_argument("--n_classes", type=int, default=2)
    parser.add_argument("--class_label", type=int, default=0)
    parser.add_argument("--guidance_scale", type=float, default=1.0)

    args = parser.parse_args()
    utils.seed_everything(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.mode in ['train', 'sample']:
        run_name = f'exps/ddpm_{args.n_dim}_{args.n_steps}_{args.lbeta}_{args.ubeta}_{args.dataset}'
        args_name = f'ddpm_{args.n_dim}_{args.n_steps}_{args.lbeta}_{args.ubeta}_{args.dataset}'
        model = DDPM(n_dim=args.n_dim, n_steps=args.n_steps)
    else:  
        run_name = f'exps/cond_ddpm_{args.n_dim}_{args.n_steps}_{args.lbeta}_{args.ubeta}_{args.dataset}_{args.n_classes}'
        args_name = f'cond_ddpm_{args.n_dim}_{args.n_steps}_{args.lbeta}_{args.ubeta}_{args.dataset}_{args.guidance_scale}'
        model = ConditionalDDPM(n_dim=args.n_dim, n_steps=args.n_steps, n_classes=args.n_classes)
    
    os.makedirs(run_name, exist_ok=True)
    noise_scheduler = NoiseScheduler(num_timesteps=args.n_steps, type="linear", beta_start=args.lbeta, beta_end=args.ubeta)
    model = model.to(device)

    if args.mode == 'train':
        epochs = args.epochs
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        data_X, _ = dataset.load_dataset(args.dataset)
        data_X = data_X.to(device)
        dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(data_X), batch_size=args.batch_size, shuffle=True)
        train(model, noise_scheduler, dataloader, optimizer, epochs, run_name)

    elif args.mode == 'train_conditional':
        epochs = args.epochs
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        data_X, data_y = dataset.load_dataset(args.dataset)
        if data_y is None:
            raise ValueError("Conditional training requires labeled data")
        data_X = data_X.to(device)
        data_y = data_y.to(device)
        dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(data_X, data_y), batch_size=args.batch_size, shuffle=True)
        trainConditional(model, noise_scheduler, dataloader, optimizer, epochs, run_name)

    elif args.mode == 'sample':
        # Unconditional sampling
        print(f"Sampling from {run_name}")
        data_X, _ = dataset.load_dataset(args.dataset)
        model.load_state_dict(torch.load(f'{run_name}/model.pth'))
        samples = sample(model, args.n_samples, noise_scheduler)
        print(f"Shape of samples tensor: {samples.shape}")
        print(f"Shape of data_X tensor: {data_X.shape}")
        print(f"NLL: {get_nll(data_X.to(device), samples.to(device))}")
        samples = samples.cpu().numpy()
        plt.figure(figsize=(6, 6))
        plt.scatter(samples[:, 0], samples[:, 1], alpha=0.6, s=1)
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.title(f"Samples for {args.dataset})")
        plt.grid()
        plt.axis('equal')
        plt.savefig(f"Samples for {args_name}.png")
        torch.save(samples, f'{run_name}/samples_{args.seed}_{args.n_samples}.pth')

    elif args.mode == 'sample_conditional':
        print(f"Conditional sampling from {run_name} for class {args.class_label}")
        data_X, data_y = dataset.load_dataset(args.dataset)
        model.load_state_dict(torch.load(f'{run_name}/model.pth'))
        samples = sampleConditional(model, args.n_samples, noise_scheduler, class_label=args.class_label)
        samples_np = samples.cpu().numpy()
        print(f"Shape of samples tensor: {samples.shape}")
        if data_y is not None:
            class_indices = (data_y == args.class_label).nonzero().squeeze()
            if len(class_indices) > 0:
                class_data = data_X[class_indices]
                print(f"Shape of class {args.class_label} data tensor: {class_data.shape}")
                print(f"Class-specific NLL: {get_nll(class_data.to(device), samples.to(device))}")
    
        plt.figure(figsize=(6, 6))
        plt.scatter(samples_np[:, 0], samples_np[:, 1], alpha=0.6, s=1)
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.title(f"Samples for Class {args_name}")
        plt.grid()
        plt.axis('equal')
        plt.savefig(f"Samples for {args_name}.png")
        
        torch.save(samples, f'{run_name}/conditional_samples_class_{args.class_label}_{args.seed}_{args.n_samples}.pth')

    elif args.mode == 'sample_multi_class':
        print(f"Multi-class conditional sampling from {run_name}")
        data_X, data_y = dataset.load_dataset(args.dataset)
        model.load_state_dict(torch.load(f'{run_name}/model.pth'))
        samples_per_class = args.n_samples // args.n_classes
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        all_samples = {}
        n_classes = args.n_classes
        n_samples_per_class = args.n_samples // args.n_classes
        colors = plt.cm.get_cmap('tab10', n_classes)
        
        plt.figure(figsize=(10, 8))
        
        for class_label in range(n_classes):
            samples = sampleConditional(
                model, 
                n_samples_per_class, 
                noise_scheduler, 
                class_label=class_label, 
                save_plot=False
            )
        
            all_samples[class_label] = samples
            samples_np = samples.cpu().numpy()
            plt.scatter(
                samples_np[:, 0], 
                samples_np[:, 1], 
                alpha=0.7, 
                s=15, 
                color=colors(class_label),
                label=f"Class {class_label}"
            )
        
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.title(f"Samples from Conditional DDPM for All Classes")
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.savefig(f"Samples for {args_name}.png")
        plt.close()
        if data_y is not None:
            for class_label in range(args.n_classes):
                class_indices = (data_y == class_label).nonzero().squeeze()
                if len(class_indices) > 0:
                    class_data = data_X[class_indices]
                    class_samples = all_samples[class_label]
                    print(f"Class {class_label}:")
                    print(f"  - Shape of ground truth data: {class_data.shape}")
                    print(f"  - Shape of generated samples: {class_samples.shape}")
                    print(f"  - Class-specific NLL: {get_nll(class_data.to(device), class_samples)}")
        all_samples_tensor = torch.cat([samples for samples in all_samples.values()], dim=0)
        torch.save(all_samples_tensor, f'{run_name}/multi_class_samples_{args.seed}_{args.n_samples}.pth')

    elif args.mode == 'sample_cfg':
        print(f"CFG sampling from {run_name} for class {args.class_label} with guidance scale {args.guidance_scale}")
        data_X, data_y = dataset.load_dataset(args.dataset)
        model.load_state_dict(torch.load(f'{run_name}/model.pth'))
        samples = sampleCFG(model, args.n_samples, noise_scheduler, guidance_scale=args.guidance_scale, class_label=args.class_label)
        print(f"Shape of samples tensor: {samples.shape}")
        if data_y is not None:
            class_indices = (data_y == args.class_label).nonzero().squeeze()
            if len(class_indices) > 0:
                class_data = data_X[class_indices]
                print(f"Shape of class {args.class_label} data tensor: {class_data.shape}")
                print(f"Class-specific NLL: {get_nll(class_data.to(device), samples.to(device))}")
        samples_np = samples.cpu().numpy()
        plt.figure(figsize=(6, 6))
        plt.scatter(samples_np[:, 0], samples_np[:, 1], alpha=0.6, s=10)
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.title(f"CFG Samples for Class {args.class_label} (scale={args.guidance_scale})")
        plt.grid()
        plt.axis('equal')
        plt.savefig(f"{run_name}/cfg_samples_class_{args.class_label}_scale_{args.guidance_scale}.png")
        
        torch.save(samples, f'{run_name}/cfg_samples_class_{args.class_label}_scale_{args.guidance_scale}_{args.seed}_{args.n_samples}.pth')
    
    else:
        raise ValueError(f"Invalid mode {args.mode}")