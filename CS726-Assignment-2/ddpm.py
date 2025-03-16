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
        self.norm1 = nn.LayerNorm(in_channels)
        self.act1 = nn.SiLU()
        self.linear1 = nn.Linear(in_channels, out_channels)
        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_channels, out_channels)
        )
        self.norm2 = nn.LayerNorm(out_channels)
        self.act2 = nn.SiLU()
        self.linear2 = nn.Linear(out_channels, out_channels)
        self.residual_connection = nn.Linear(in_channels, out_channels) if in_channels != out_channels else nn.Identity()
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
        residual = self.residual_connection(x)
        feature = self.norm1(x)
        feature = self.act1(feature)
        feature = self.linear1(feature)
        time_projection = self.time_proj(time_emb)
        feature = feature + time_projection
        feature = self.norm2(feature)
        feature = self.act2(feature)
        feature = self.linear2(feature)
        if self.use_attention:
            feature = feature + self.attention(self.attention_norm(feature))
        return feature + residual

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
        batch_size = x.shape[0]
        q = self.query(x).view(batch_size, self.num_heads, self.head_dim)
        k = self.key(x).view(batch_size, self.num_heads, self.head_dim)
        v = self.value(x).view(batch_size, self.num_heads, self.head_dim)
        attention = torch.einsum('bhd,bhd->bh', q, k)
        attention = F.softmax(attention, dim=-1)
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
        self.time_embedding = SinusoidalPositionEmbeddings(time_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim)
        )
        self.input_proj = nn.Linear(n_dim, model_channels)
        self.down_blocks = nn.ModuleList()
        current_channels = model_channels
        down_channel_list = []
        for mult in channel_mults:
            out_channels = model_channels * mult
            self.down_blocks.append(
                ResidualBlock(current_channels, out_channels, time_emb_dim, use_attention=(mult >= 4))
            )
            down_channel_list.append(current_channels)
            current_channels = out_channels
        self.middle_block1 = ResidualBlock(current_channels, current_channels, time_emb_dim, use_attention=True)
        self.middle_block2 = ResidualBlock(current_channels, current_channels, time_emb_dim, use_attention=False)
        self.up_blocks = nn.ModuleList()   
        for mult in reversed(channel_mults):
            out_channels = model_channels * mult
            skip_channels = down_channel_list.pop()
            self.up_blocks.append(
                ResidualBlock(current_channels + skip_channels, out_channels, time_emb_dim, use_attention=(mult >= 4))
            )
            current_channels = out_channels
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
        time_emb = self.time_embedding(t)
        time_emb = self.time_mlp(time_emb)
        feature = self.input_proj(x)
        skips = []
        for down_block in self.down_blocks:
            skips.append(feature)
            feature = down_block(feature, time_emb)
        feature = self.middle_block1(feature, time_emb)
        feature = self.middle_block2(feature, time_emb)
        for up_block in self.up_blocks:
            skip = skips.pop()
            feature = torch.cat([feature, skip], dim=-1)
            feature = up_block(feature, time_emb)
        feature = self.norm_out(feature)
        feature = self.act_out(feature)
        feature = self.out_proj(feature)
        return feature
    
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
        elif type == "cosine":
            self.init_cosine_schedule(**kwargs)
        elif type == "sigmoid":
            self.init_sigmoid_schedule(**kwargs)
        else:
            raise NotImplementedError(f"{type} scheduler is not implemented")


    def init_linear_schedule(self, beta_start, beta_end):
        """
        Precompute whatever quantities are required for training and sampling
        """
        self.betas = torch.linspace(beta_start, beta_end, self.num_timesteps, dtype=torch.float32)
        self.alphas = 1.0 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, 0)

    def init_cosine_schedule(self, beta_start, beta_end, s=0.008):
        """
        Precompute whatever quantities are required for training and sampling with cosine noise schedule
        """

        steps = torch.linspace(0, self.num_timesteps, self.num_timesteps + 1, dtype=torch.float32)
        alphas_bar = torch.cos(((steps / self.num_timesteps) + s) / (1 + s) * (torch.acos(torch.tensor(-1.0)) / 2)) ** 2
        self.alpha_bar = alphas_bar / alphas_bar[0]
        self.alphas = self.alpha_bar[1:] / self.alpha_bar[:-1]
        self.betas = 1.0 - self.alphas
        self.betas = torch.clip(self.betas, beta_start, beta_end)
        self.alphas = 1.0 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, 0)

    def init_sigmoid_schedule(self, beta_start=0.0001, beta_end=0.02, k=10):
        """
        Precompute whatever quantities are required for training and sampling with sigmoid noise schedule
        """
        steps = torch.linspace(-k, k, self.num_timesteps, dtype=torch.float32)
        sigmoid = torch.sigmoid(steps)
        self.betas = beta_start + (beta_end - beta_start) * sigmoid
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
    def __init__(self, n_classes = 2, n_dim=3, n_steps=200):
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

    def predict_proba(self, x, num_noise_samples=10, timesteps=None):
        batch_size = x.shape[0]
        n_classes = self.model.n_classes
        device = self.device
        x = x.to(device)
        if timesteps is None:
            timesteps = [self.noise_scheduler.num_timesteps // 4, 
                        self.noise_scheduler.num_timesteps // 2,
                        3 * self.noise_scheduler.num_timesteps // 4]
        
        class_scores = torch.zeros((batch_size, n_classes), device=device)
        for t_step in timesteps:
            t = torch.full((batch_size,), t_step, dtype=torch.long, device=device)
            alpha_bar_t = self.noise_scheduler.alpha_bar[t].view(-1, 1)
            alpha_bar_t = alpha_bar_t.to(device)
            for _ in range(num_noise_samples):
                noise = torch.randn_like(x, device=device)
                x_t = torch.sqrt(alpha_bar_t) * x + torch.sqrt(1 - alpha_bar_t) * noise
                
                for c in range(n_classes):
                    class_labels = torch.full((batch_size,), c, dtype=torch.long, device=device)
                    predicted_noise = self.model(x_t, t, class_labels)
                    error = torch.mean((predicted_noise - noise)**2, dim=1)
                    class_scores[:, c] -= error  
        class_scores /= (len(timesteps) * num_noise_samples)
        temperature = 10.0  
        probs = F.softmax(class_scores * temperature, dim=1)
        return probs

class Classifier(nn.Module):
    def __init__(self, n_dim=3, n_classes=10):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_classes)
        )
    def forward(self, x):
        return self.network(x)
    
    def predict(self, x):
        logits = self.forward(x)
        return torch.argmax(logits, dim=1)
    
    def predict_proba(self, x):
        logits = self.forward(x)
        return F.softmax(logits, dim=1)
    
def train_classifier(model, train_loader, test_loader, n_epochs=30, lr=1e-3):
    device = next(model.parameters()).device
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        train_acc = 100. * correct / total
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        test_acc = 100. * correct / total
        
        print(f'Epoch: {epoch+1}/{n_epochs} | Train Loss: {train_loss/len(train_loader):.3f} | Train Acc: {train_acc:.3f}% | Test Loss: {test_loss/len(test_loader):.3f} | Test Acc: {test_acc:.3f}%')
    
    return model

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
    x_t = torch.randn((n_samples, n_dim), device=device)
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
    plt.savefig("images/moon.png")
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
    x_t = torch.randn((n_samples, n_dim), device=device) 
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
        plt.savefig(f"images/conditional_samples_class_{class_label}.png")
        
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

@torch.no_grad()
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
        noise_pred = -guidance_scale * noise_pred_unconditional + (1 + guidance_scale) * noise_pred_conditional
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

@torch.no_grad()
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


def evaluate_with_guidance_scales(model, classifier, dataset, guidance_scales=[0, 1, 2, 5, 10]):
    results = {}
    for scale in guidance_scales:
        print(f"Evaluating with guidance scale: {scale}")
        samples = []
        labels = []
        for class_label in range(model.n_classes):
            class_samples = sampleConditional(
                model, 
                n_samples=100,
                class_label=class_label,
                guidance_scale=scale
            )
            samples.append(class_samples)
            labels.extend([class_label] * 100)
        samples = torch.cat(samples, dim=0)
        labels = torch.tensor(labels, device=samples.device)
        with torch.no_grad():
            predicted = classifier(samples).argmax(dim=1)
            accuracy = (predicted == labels).float().mean().item()
        
        results[scale] = {
            "accuracy": accuracy,
        }
        
        print(f"Guidance scale {scale}: Classifier accuracy = {accuracy:.4f}")
    
    return results


def compare_classifiers(standard_classifier, ddpm_classifier, test_loader):
    device = next(standard_classifier.parameters()).device
    
    standard_correct = 0
    ddpm_correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            total += targets.size(0)
            standard_preds = standard_classifier.predict(inputs)
            standard_correct += (standard_preds == targets).sum().item()
            ddpm_preds = ddpm_classifier.predict(inputs)
            ddpm_correct += (ddpm_preds == targets).sum().item()
    
    standard_acc = 100. * standard_correct / total
    ddpm_acc = 100. * ddpm_correct / total
    
    print(f"Standard Classifier Accuracy: {standard_acc:.2f}%")
    print(f"DDPM Classifier Accuracy: {ddpm_acc:.2f}%")
    
    return standard_acc, ddpm_acc


def evaluate_cfg_sampling(model, classifier, noise_scheduler, guidance_scales=[0, 1, 2, 5, 10], n_samples=100):
    """
    Evaluate the quality of samples generated with classifier-free guidance using a classifier
    
    Args:
        model: ConditionalDDPM model
        classifier: Trained classifier model
        noise_scheduler: NoiseScheduler
        guidance_scales: List of guidance scale values to test
        n_samples: Number of samples per class
        
    Returns:
        dict: Results for each guidance scale, containing accuracy metrics
    """
    results = {}
    device = next(model.parameters()).device
    
    for scale in guidance_scales:
        print(f"Evaluating with guidance scale: {scale}")
        samples = []
        labels = []
        for class_label in range(model.n_classes):
            class_samples = sampleCFG(
                model, 
                n_samples=n_samples,
                noise_scheduler=noise_scheduler,
                guidance_scale=scale,
                class_label=class_label
            )
            samples.append(class_samples)
            labels.extend([class_label] * n_samples)
        

        samples = torch.cat(samples, dim=0)
        labels = torch.tensor(labels, device=device)
        

        with torch.no_grad():
            logits = classifier(samples)
            predicted = logits.argmax(dim=1)
            accuracy = (predicted == labels).float().mean().item()

            class_accuracies = {}
            for class_label in range(model.n_classes):
                class_mask = (labels == class_label)
                if class_mask.sum() > 0:
                    class_acc = (predicted[class_mask] == class_label).float().mean().item()
                    class_accuracies[f"class_{class_label}_accuracy"] = class_acc
        
        results[scale] = {
            "overall_accuracy": accuracy,
            **class_accuracies
        }
        
        print(f"Guidance scale {scale}: Overall accuracy = {accuracy:.4f}")
        for class_label in range(model.n_classes):
            print(f"  Class {class_label} accuracy: {class_accuracies.get(f'class_{class_label}_accuracy', 0):.4f}")
    
    return results

def train_and_evaluate_cfg(model, noise_scheduler, dataset_name, guidance_scales=[0, 1, 2, 5, 10], n_samples=100, classifier_epochs=30, save_path='cfg_evaluation'):
    """
    Train a classifier on the dataset, generate samples with different guidance scales,
    and evaluate how well the classifier recognizes the generated samples.
    
    Args:
        model: ConditionalDDPM model
        noise_scheduler: NoiseScheduler
        dataset_name: Name of dataset to load
        guidance_scales: List of guidance scale values to test
        n_samples: Number of samples per class to generate
        classifier_epochs: Number of epochs to train the classifier
        save_path: Directory to save results and plots
    
    Returns:
        tuple: (trained classifier, evaluation results)
    """
    os.makedirs(save_path, exist_ok=True)
    device = next(model.parameters()).device
    print("Loading dataset...")
    data_X, data_y = dataset.load_dataset(dataset_name)
    data_X = data_X.to(device)
    data_y = data_y.to(device)
    n_dim = data_X.shape[1]
    n_classes = model.n_classes
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        data_X.cpu().numpy(), data_y.cpu().numpy(), test_size=0.2, random_state=42
    )
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.long).to(device)
    y_test = torch.tensor(y_test, dtype=torch.long).to(device)
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
    print("Training classifier...")
    classifier = Classifier(n_dim=n_dim, n_classes=n_classes).to(device)
    classifier = train_classifier(classifier, train_loader, test_loader, n_epochs=classifier_epochs)
    torch.save(classifier.state_dict(), f"{save_path}/classifier.pth")
    print(f"Classifier saved to {save_path}/classifier.pth")

    def sample_and_evaluate_cfg(guidance_scale):
        print(f"\nGenerating samples with guidance scale: {guidance_scale}")
        all_samples = []
        all_labels = []
        for class_label in range(n_classes):
            print(f"  Generating samples for class {class_label}...")
            samples = sampleCFG(
                model=model,
                n_samples=n_samples,
                noise_scheduler=noise_scheduler,
                guidance_scale=guidance_scale,
                class_label=class_label
            )
            all_samples.append(samples)
            all_labels.extend([class_label] * n_samples)
            samples_np = samples.cpu().numpy()
            if n_dim == 2:  
                plt.figure(figsize=(6, 6))
                plt.scatter(samples_np[:, 0], samples_np[:, 1], alpha=0.7, s=10)
                plt.title(f"Class {class_label}, Guidance Scale {guidance_scale}")
                plt.xlabel("x1")
                plt.ylabel("x2")
                plt.axis('equal')
                plt.grid(True)
                plt.savefig(f"{save_path}/class{class_label}_scale{guidance_scale}.png")
                plt.close()
        all_samples = torch.cat(all_samples, dim=0)
        all_labels = torch.tensor(all_labels, dtype=torch.long, device=device)
        torch.save(all_samples, f"{save_path}/samples_scale{guidance_scale}.pt")
        torch.save(all_labels, f"{save_path}/labels_scale{guidance_scale}.pt")
        classifier.eval()
        with torch.no_grad():
            logits = classifier(all_samples)
            predicted = torch.argmax(logits, dim=1)
            accuracy = (predicted == all_labels).float().mean().item()
            class_accuracies = {}
            confusion_matrix = torch.zeros(n_classes, n_classes, device=device)
            for i in range(len(all_labels)):
                confusion_matrix[all_labels[i], predicted[i]] += 1      
            for c in range(n_classes):
                class_mask = (all_labels == c)
                if class_mask.sum() > 0:
                    class_acc = (predicted[class_mask] == c).float().mean().item()
                    class_accuracies[f"class_{c}_accuracy"] = class_acc
            for i in range(n_classes):
                confusion_matrix[i] = confusion_matrix[i] / confusion_matrix[i].sum() * 100
        print(f"Results for guidance scale {guidance_scale}:")
        print(f"  Overall accuracy: {accuracy:.4f}")
        for c in range(n_classes):
            print(f"  Class {c} accuracy: {class_accuracies.get(f'class_{c}_accuracy', 0):.4f}")
        if n_classes > 2:
            plt.figure(figsize=(8, 6))
            cm = confusion_matrix.cpu().numpy()
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title(f'Confusion Matrix - Scale {guidance_scale}')
            plt.colorbar()
            tick_marks = range(n_classes)
            plt.xticks(tick_marks, range(n_classes))
            plt.yticks(tick_marks, range(n_classes))
            plt.xlabel('Predicted Class')
            plt.ylabel('True Class')
            for i in range(n_classes):
                for j in range(n_classes):
                    plt.text(j, i, f'{cm[i, j]:.1f}',
                            horizontalalignment="center",
                            color="white" if cm[i, j] > 50 else "black")
            
            plt.savefig(f"{save_path}/confusion_matrix_scale{guidance_scale}.png")
            plt.close()
            
        return {
            "overall_accuracy": accuracy,
            **class_accuracies,
            "confusion_matrix": confusion_matrix.cpu().numpy().tolist()
        }
    
    results = {}
    for scale in guidance_scales:
        results[scale] = sample_and_evaluate_cfg(scale)
    plt.figure(figsize=(10, 6))
    scales = list(results.keys())
    accuracies = [results[s]["overall_accuracy"] for s in scales]
    
    plt.plot(scales, accuracies, marker='o', linestyle='-', linewidth=2)
    plt.xlabel('Guidance Scale')
    plt.ylabel('Classification Accuracy (%)')
    plt.title('Effect of Guidance Scale on Sample Quality')
    plt.grid(True)
    plt.xticks(scales)
    plt.ylim(0, 1.05)
    if n_classes > 1:
        for c in range(n_classes):
            class_accs = [results[s].get(f"class_{c}_accuracy", 0) for s in scales]
            plt.plot(scales, class_accs, marker='s', linestyle='--', label=f'Class {c}')
        plt.legend()
    
    plt.savefig(f"{save_path}/accuracy_vs_scale.png")
    plt.close()
    import json
    with open(f"{save_path}/evaluation_results.json", 'w') as f:
        json.dump({str(k): v for k, v in results.items()}, f, indent=2)
    
    print(f"Evaluation complete. Results saved to {save_path}/")
    return classifier, results


import torch.nn.functional as F

def classifier_reward(samples, classifier, target_class):
    classifier.eval()
    with torch.no_grad():
        probs = classifier.predict_proba(samples) 
        reward = probs[:, target_class]

    return reward


def compute_svdd_accuracy(standard_classifier, all_samples, true_labels):
    device = next(standard_classifier.parameters()).device
    standard_classifier.eval()
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for class_label, samples in all_samples.items():
            samples = samples.to(device)
            outputs = standard_classifier(samples)
            predicted_labels = torch.argmax(outputs, dim=1)
            
            true_class_labels = torch.full((samples.shape[0],), class_label, dtype=torch.long, device=device)
            correct += (predicted_labels == true_class_labels).sum().item()
            total += samples.shape[0]
    
    accuracy = 100 * correct / total if total > 0 else 0
    return accuracy

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=['train', 'sample', 'train_conditional', 'sample_conditional', 'sample_cfg', 'sample_multi_class', 'compare_classifiers', 'sampleSVDD'], default='sample')
    parser.add_argument("--noise_schedule", choices=['linear', 'cosine', 'sigmoid'], default='linear')
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

    if args.mode in ['train', 'sample', 'sampleSVDD']:
        run_name = f'exps/ddpm_{args.n_dim}_{args.n_steps}_{args.lbeta}_{args.ubeta}_{args.dataset}_{args.noise_schedule}'
        args_name = f'ddpm_{args.n_dim}_{args.n_steps}_{args.lbeta}_{args.ubeta}_{args.dataset}_{args.noise_schedule}'
        model = DDPM(n_dim=args.n_dim, n_steps=args.n_steps)
    else:  
        run_name = f'exps/cond_ddpm_{args.n_dim}_{args.n_steps}_{args.lbeta}_{args.ubeta}_{args.dataset}_{args.n_classes}_{args.noise_schedule}'
        args_name = f'cond_ddpm_{args.n_dim}_{args.n_steps}_{args.lbeta}_{args.ubeta}_{args.dataset}_{args.guidance_scale}_{args.noise_schedule}'
        model = ConditionalDDPM(n_dim=args.n_dim, n_steps=args.n_steps, n_classes=args.n_classes)
    
    os.makedirs(run_name, exist_ok=True)
    if (args.noise_schedule == 'linear'):
        noise_scheduler = NoiseScheduler(num_timesteps=args.n_steps, type="linear", beta_start=args.lbeta, beta_end=args.ubeta)
    elif (args.noise_schedule == 'cosine'):
        noise_scheduler = NoiseScheduler(num_timesteps=args.n_steps, type="cosine", beta_start=args.lbeta, beta_end=args.ubeta)
    elif (args.noise_schedule == 'sigmoid'):
        noise_scheduler = NoiseScheduler(num_timesteps=args.n_steps, type="sigmoid", beta_start=args.lbeta, beta_end=args.ubeta)
    else:
        raise ValueError("Invalid noise schedule type")
    model = model.to(device)

    if args.mode == 'train':
        epochs = args.epochs
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        data_X, _ = dataset.load_dataset(args.dataset)
        samples = data_X
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
        print(f"Sampling from {run_name}")
        data_X, _ = dataset.load_dataset(args.dataset)
        model.load_state_dict(torch.load(f'{run_name}/model.pth'))
        samples = sample(model, args.n_samples, noise_scheduler)
        print(f"NLL: {get_nll(data_X.to(device), samples.to(device))}")
        samples = samples.cpu().numpy()
        if args.n_dim == 3:
            from mpl_toolkits.mplot3d import Axes3D
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111, projection='3d')

            ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2], alpha=0.6, s=1)

            ax.set_xlabel("x1")
            ax.set_ylabel("x2")
            ax.set_zlabel("x3")
            ax.set_title(f"3D Samples for Class {args_name}")
            plt.savefig(f"images/Samples for {args_name}.png")
            exit(0)
        plt.figure(figsize=(6, 6))
        plt.scatter(samples[:, 0], samples[:, 1], alpha=0.6, s=1)
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.title(f"Samples for {args.dataset}")
        plt.grid()
        plt.axis('equal')
        plt.savefig(f"images/Samples for {args_name}.png")
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
        plt.savefig(f"images/Samples for {args_name}.png")
        
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
        plt.savefig(f"images/Samples for {args_name}.png")
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
        classifier, results = train_and_evaluate_cfg(
            model=model,
            noise_scheduler=noise_scheduler,
            dataset_name=args.dataset,
            guidance_scales=args.guidance_scales,
            n_samples=args.n_samples,
            classifier_epochs=args.classifier_epochs,
            save_path=args.save_path
        ) 
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
            samples = sampleCFG(model, args.n_samples, noise_scheduler, guidance_scale=args.guidance_scale, class_label=class_label)
        
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
        plt.savefig(f"images/Samples - CFG for {args_name}.png")
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
        
        torch.save(all_samples_tensor, f'{run_name}/cfg_samples_class_{args.class_label}_scale_{args.guidance_scale}_{args.seed}_{args.n_samples}.pth')

    elif args.mode == 'compare_classifiers':
        print(f"Comparing standard classifier and DDPM classifier on {args.dataset}")
        data_X, data_y = dataset.load_dataset(args.dataset)
        data_X = data_X.to(device)
        data_y = data_y.to(device)
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            data_X.cpu().numpy(), data_y.cpu().numpy(), test_size=0.2, random_state=args.seed
        )
        X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
        X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_train = torch.tensor(y_train, dtype=torch.long).to(device)
        y_test = torch.tensor(y_test, dtype=torch.long).to(device)
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        print("Training standard classifier...")
        standard_classifier = Classifier(n_dim=args.n_dim, n_classes=args.n_classes).to(device)
        standard_classifier = train_classifier(standard_classifier, train_loader, test_loader, n_epochs=args.epochs)
        print("Setting up DDPM classifier...")
        model_path = f'{run_name}/model.pth'
        
        if os.path.exists(model_path):
            print(f"Loading DDPM model from {model_path}")
            model.load_state_dict(torch.load(model_path))
        else:
            print(f"Training DDPM model (model not found at {model_path})")
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(data_X, data_y), 
                                                    batch_size=args.batch_size, shuffle=True)
            trainConditional(model, noise_scheduler, dataloader, optimizer, args.epochs, run_name)
        ddpm_classifier = ClassifierDDPM(model, noise_scheduler)
        print("Comparing classifier performance...")
        standard_acc, ddpm_acc = compare_classifiers(standard_classifier, ddpm_classifier, test_loader)
        results = {
            "standard_classifier_accuracy": standard_acc,
            "ddpm_classifier_accuracy": ddpm_acc,
            "dataset": args.dataset,
            "n_dim": args.n_dim,
            "n_classes": args.n_classes,
            "n_steps": args.n_steps,
            "noise_schedule": args.noise_schedule
        }
        import json
        results_path = f"{run_name}/classifier_comparison_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"Results saved to {results_path}")
        labels = ['Standard Classifier', 'DDPM Classifier']
        accuracies = [standard_acc, ddpm_acc]
        plt.figure(figsize=(8, 5))
        plt.bar(labels, accuracies, color=['blue', 'orange'])
        plt.ylabel('Accuracy (%)')
        plt.title(f'Classifier Comparison on {args.dataset} Dataset')
        plt.ylim(0, 100)
        for i, v in enumerate(accuracies):
            plt.text(i, v + 1, f"{v:.2f}%", ha='center')
        
        plt.savefig(f"{run_name}/classifier_comparison.png")
        print(f"Comparison plot saved to {run_name}/classifier_comparison.png")

    elif args.mode == 'sampleSVDD':
        data_X, data_y = dataset.load_dataset(args.dataset)
        data_X = data_X.to(device)
        data_y = data_y.to(device)
        model.load_state_dict(torch.load(f'{run_name}/model.pth'))
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            data_X.cpu().numpy(), data_y.cpu().numpy(), test_size=0.2, random_state=args.seed
        )
        X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
        X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_train = torch.tensor(y_train, dtype=torch.long).to(device)
        y_test = torch.tensor(y_test, dtype=torch.long).to(device)
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        print("Training standard classifier...")
        standard_classifier = Classifier(n_dim=args.n_dim, n_classes=args.n_classes).to(device)
        standard_classifier = train_classifier(standard_classifier, train_loader, test_loader, n_epochs=args.epochs)
        print("Setting up DDPM classifier...")
        samples_per_class = args.n_samples // args.n_classes
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        all_samples = {}
        n_classes = args.n_classes
        n_samples_per_class = args.n_samples // args.n_classes
        colors = plt.cm.get_cmap('tab10', n_classes)
        
        plt.figure(figsize=(10, 8))
        
        for class_label in range(n_classes):
            reward_fn = lambda x: classifier_reward(x, standard_classifier, class_label)
            reward_scale = 0.5
            samples = sampleSVDD(model, args.n_samples, noise_scheduler, reward_scale, reward_fn)
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
        all_samples_tensor = torch.cat([samples for samples in all_samples.values()], dim=0)
        true_labels = torch.cat([torch.full((samples.shape[0],), class_label, dtype=torch.long, device=device) for class_label, samples in all_samples.items()], dim=0)
        svdd_accuracy = compute_svdd_accuracy(standard_classifier, all_samples, true_labels)
        print(f"SVDD Sample Classification Accuracy: {svdd_accuracy:.2f}%")

        torch.save(all_samples_tensor, f'{run_name}/svdd_samples_class_{args.class_label}_scale_{args.guidance_scale}_{args.seed}_{args.n_samples}.pth')

    
    else:
        raise ValueError(f"Invalid mode {args.mode}")