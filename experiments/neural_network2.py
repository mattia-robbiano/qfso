import os
import pickle
from itertools import combinations
from typing import Tuple

import numpy as np
from scipy.stats import norm
from numba import njit
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def setup_ddp() -> int:
    """Inizializza l'ambiente distribuito se lanciato con torchrun."""
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return local_rank
    return 0

def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def load_real_spoofing_data(N: int, filepath: str, mmd_sigma: float):
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
        
    target_wht_trunc = data['expvals']
    
    ops = []
    for weight in (1, 2):
        for idxs in combinations(range(N), weight):
            op = np.zeros(N, dtype=np.float32)
            op[list(idxs)] = 1.0
            ops.append(op)
    K_matrix = np.array(ops, dtype=np.float32)
    
    hw_array = K_matrix.sum(axis=1)
    weights = np.array([mmd_kernel_weight(int(h), N, mmd_sigma) for h in hw_array], dtype=np.float32)
    
    target_wht_tensor = torch.from_numpy(target_wht_trunc).float()
    K_matrix_tensor = torch.from_numpy(K_matrix)
    weights_tensor = torch.from_numpy(weights)
    
    return target_wht_tensor, K_matrix_tensor, weights_tensor

@njit(cache=True)
def wht_numba(p: np.ndarray) -> np.ndarray:
    a = p.astype(np.float64).copy()
    h = 1
    while h < a.size:
        for i in range(0, a.size, h * 2):
            for j in range(i, i + h):
                x = a[j]
                y = a[j + h]
                a[j] = x + y
                a[j + h] = x - y
        h *= 2
    return a

@njit(cache=True)
def mmd_kernel_weight(hw: int, n: int, sigma: float) -> float:
    p = 0.5 * (1.0 - np.exp(-1.0 / (2.0 * sigma)))
    return (p**hw) * ((1.0 - p) ** (n - hw))


class MMD(nn.Module):
    """Contenitore per i tensori target (il calcolo effettivo avviene nel Trainer)"""
    def __init__(self, N: int, target_wht: torch.Tensor, K_matrix: torch.Tensor, kernel_weights: torch.Tensor):
        super().__init__()
        self.N = N
        self.register_buffer('target_wht', target_wht.float())
        self.register_buffer('K_matrix', K_matrix.float())
        self.register_buffer('kernel_weights', kernel_weights.float())


class Model(nn.Module):
    """
    Deep Residual MLP Model per spoofing ad alta capacità.
    """
    def __init__(self, N: int, hidden_dim: int = 1024):
        super().__init__()
        self.N = N
        self.hidden_dim = hidden_dim
        
        # Proiezione iniziale
        self.input_layer = nn.Sequential(
            nn.Linear(N, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # Blocchi Residuali per sostenere i gradienti dello STE
        self.res_block1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        self.res_block2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # Livello di output per i logit dei bit
        self.output_layer = nn.Linear(hidden_dim, N)
        self.relu = nn.ReLU()
        
    def forward(self, z: torch.Tensor, temp: float = 1.0, hard: bool = True) -> torch.Tensor:
        h = self.input_layer(z)
        # Skip connections
        h = self.relu(h + self.res_block1(h))
        h = self.relu(h + self.res_block2(h))
        
        logits = self.output_layer(h)
        # Temperature scaling per ridurre il bias dello STE
        probs = torch.sigmoid(logits / temp)
        
        if hard:
            samples = (probs > 0.5).float()
            samples = samples.detach() - probs.detach() + probs
            return samples
            
        return probs
            
    def get_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
    
    def get_config(self) -> dict:
        return {
            "N": self.N,
            "hidden_dim": self.hidden_dim,
            "actual_params": self.get_param_count(),
        }


class Trainer:
    def __init__(
        self,
        model: Model,
        target_wht: torch.Tensor,
        K_matrix: torch.Tensor,
        kernel_weights: torch.Tensor,
        device: torch.device,
        local_rank: int = 0,
        lr: float = 1e-3,
        total_epochs: int = 10000,
    ):
        self.device = device
        self.local_rank = local_rank
        self.N = model.N
        self.is_main_process = (local_rank == 0)
        
        model = model.to(device)
        if dist.is_initialized():
            self.model = DDP(model, device_ids=[local_rank])
            self.base_model = self.model.module
        else:
            self.model = model
            self.base_model = model
            
        self.mmd_loss = MMD(self.N, target_wht, K_matrix, kernel_weights).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=total_epochs)
        self.history = {"loss": [], "lr": []}
    
    def train_step(self, batch_size: int = 1024, num_frequencies: int = 2048, current_epoch: int = 0, total_epochs: int = 10000) -> float:
        self.optimizer.zero_grad()
        
        # Batch reale massiccio in un solo passaggio per distruggere il bias di varianza
        z = torch.randn(batch_size, self.N, device=self.device)
        
        # Passiamo temp=1.0 per usare il modello liscio senza annealing
        samples = self.model(z, temp=1.0, hard=False)
        
        spins = 1.0 - 2.0 * samples
        spins_expanded = spins.unsqueeze(1) # (batch_size, 1, N)
        
        M_total = self.mmd_loss.target_wht.size(0)
        
        idx = torch.randint(0, M_total, (num_frequencies,), device=self.device)
        
        K_sampled = self.mmd_loss.K_matrix[idx]            
        target_sampled = self.mmd_loss.target_wht[idx]     
        weight_sampled = self.mmd_loss.kernel_weights[idx] 
        
        K_expanded = K_sampled.unsqueeze(0)                
        term = spins_expanded * K_expanded + (1.0 - K_expanded)
        
        wht_batch = term.prod(dim=2)                       
        fourier_emp = wht_batch.mean(dim=0)                
        diff = fourier_emp - target_sampled
        
        loss = torch.sum(weight_sampled * (diff ** 2)) * (M_total / num_frequencies)
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        total_loss_val = loss.item()
        
        if dist.is_initialized():
            loss_tensor = torch.tensor(total_loss_val, device=self.device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            total_loss_val = (loss_tensor / dist.get_world_size()).item()
            
        if self.is_main_process:
            self.history["loss"].append(total_loss_val)
            self.history["lr"].append(self.optimizer.param_groups[0]["lr"])
        
        return total_loss_val
    
    def train(self, num_epochs: int = 10000, batch_size: int = 256, num_frequencies: int = 2048, eval_interval: int = 100):
        if self.is_main_process:
            print("Starting training...")
            print(f"{'Epoch':<8} {'Loss':<12}")
            print("-" * 22)
        
        for epoch in range(num_epochs):
            loss = self.train_step(
                batch_size=batch_size, 
                num_frequencies=num_frequencies,
                current_epoch=epoch,
                total_epochs=num_epochs
            )
            
            self.scheduler.step()
            
            if self.is_main_process and (epoch + 1) % eval_interval == 0:
                print(f"{epoch + 1:<8} {loss:.6f}")
        
        if self.is_main_process:
            print("-" * 22)
            print("Training complete!")
            
    def get_history(self) -> dict:
        return self.history

if __name__ == "__main__":
    local_rank = setup_ddp()
    is_main_process = (local_rank == 0)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    N = 484
    epochs = 10000

    model = Model(N=N, hidden_dim=1024)

    if is_main_process:
        config = model.get_config()
        print("=" * 60)
        print("Model Configuration (Deep ResNet)")
        print("=" * 60)
        print(f"Bitstring length (N):        {config['N']}")
        print(f"Hidden dimension (H):        {config['hidden_dim']}")
        print(f"Actual parameters:           {config['actual_params']}")
        print()
    
    filepath = 'expvals_IqpSimulator_dwave.pkl'
    target_wht_tensor, K_matrix_tensor, weights_tensor = load_real_spoofing_data(
        N=N, filepath=filepath, mmd_sigma=0.1 * N
    )
    
    trainer = Trainer(
        model=model,
        target_wht=target_wht_tensor,
        K_matrix=K_matrix_tensor,
        kernel_weights=weights_tensor,
        device=device,
        local_rank=local_rank,
        lr=1e-3,
        total_epochs=epochs 
    )
    
    if is_main_process:
        print("=" * 60)
        print("Training")
        print("=" * 60)
        print()
    
    trainer.train(
        num_epochs=epochs,
        batch_size=512,          
        num_frequencies=2048,    
        eval_interval=100
    )
    
    if is_main_process:
        min_loss = np.min(trainer.get_history()["loss"])
        print(f"Addestramento concluso. Minima Loss MMD raggiunta: {min_loss:.4e}")

        model_save_path = "spoofing_model_weights.pth"
        torch.save(trainer.base_model.state_dict(), model_save_path)
        print(f"Model weights successfully saved to: {model_save_path}")

        history_save_path = "training_history.pkl"
        with open(history_save_path, "wb") as f:
            pickle.dump(trainer.get_history(), f)
        print(f"Training history successfully saved to: {history_save_path}")
        print("=" * 60)

    cleanup_ddp()