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
        attention = torch.einsum('bhd,bhd->bh', q, k) / (self.head_dim ** 0.5)
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

class ConditionalDDPM():
    pass
    
class ClassifierDDPM():
    """
    ClassifierDDPM implements a classification algorithm using the DDPM model
    """
    
    def __init__(self, model: ConditionalDDPM, noise_scheduler: NoiseScheduler):
        pass

    def __call__(self, x):
        pass

    def predict(self, x):
        pass

    def predict_proba(self, x):
        pass

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
        # print(t_tensor)
        noise_pred = model(x_t, t_tensor)
        # print(noise_pred)
        alpha_bar_t = noise_scheduler.alpha_bar[t]
        alpha_t = noise_scheduler.alphas[t]
        beta_t = noise_scheduler.betas[t]
        # print(alpha_bar_t, alpha_t, beta_t)
        mean = (1 / torch.sqrt(alpha_t)) * (x_t - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * noise_pred)
        # print(t, mean, flush=True)
        if t > 1:
            noise = torch.randn_like(x_t, device=device)
            std = torch.sqrt(beta_t)
            x_t = mean + std * noise
        else:
            x_t = mean
        # print(t, x_t, flush=True)
        
        if return_intermediate:
            intermediate_steps.append(x_t.clone().detach())
    
    # print(x_t)

    x_t_np = x_t.cpu().numpy()
    # print(x_t_np)
    # Scatter plot
    plt.figure(figsize=(6, 6))
    plt.scatter(x_t_np[:, 0], x_t_np[:, 1], alpha=0.6)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Final Sampled Points")
    plt.grid()
    plt.savefig("moon.png")
    return (x_t if not return_intermediate else intermediate_steps)

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
    pass

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
    pass
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=['train', 'sample'], default='sample')
    parser.add_argument("--n_steps", type=int, default=None)
    parser.add_argument("--lbeta", type=float, default=None)
    parser.add_argument("--ubeta", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--n_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--dataset", type=str, default = None)
    parser.add_argument("--seed", type=int, default = 42)
    parser.add_argument("--n_dim", type=int, default = None)

    args = parser.parse_args()
    utils.seed_everything(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    run_name = f'exps/ddpm_{args.n_dim}_{args.n_steps}_{args.lbeta}_{args.ubeta}_{args.dataset}' # can include more hyperparams
    os.makedirs(run_name, exist_ok=True)

    model = DDPM(n_dim=args.n_dim, n_steps=args.n_steps)
    noise_scheduler = NoiseScheduler(num_timesteps=args.n_steps, beta_start=args.lbeta, beta_end=args.ubeta)
    model = model.to(device)

    if args.mode == 'train':
        epochs = args.epochs
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        data_X, data_y = dataset.load_dataset(args.dataset)
        # can split the data into train and test -- for evaluation later
        data_X = data_X.to(device)
        if data_y != None:
            data_y = data_y.to(device)
        if data_y == None:
            dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(data_X), batch_size=args.batch_size, shuffle=True)
        else:
            dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(data_X, data_y), batch_size=args.batch_size, shuffle=True)
        train(model, noise_scheduler, dataloader, optimizer, epochs, run_name)

    elif args.mode == 'sample':
        model.load_state_dict(torch.load(f'{run_name}/model.pth'))
        samples = sample(model, args.n_samples, noise_scheduler)
        torch.save(samples, f'{run_name}/samples_{args.seed}_{args.n_samples}.pth')
    else:
        raise ValueError(f"Invalid mode {args.mode}")