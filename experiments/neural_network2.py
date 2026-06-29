"""
Differentiable generative model for bitstring distributions.
MMD² loss in Fourier space with proper gradient flow via Straight-Through Estimator.
Parameter count strictly controlled: P = N^k * r
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.stats import norm
from numba import njit
from typing import Tuple


# ============================================================================
# Walsh-Hadamard Transform (Fourier basis)
# ============================================================================

@njit(cache=True)
def wht_numba(p: np.ndarray) -> np.ndarray:
    """Fast Walsh-Hadamard transform (unnormalized)."""
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
    """Binomial MMD kernel weight."""
    p = 0.5 * (1.0 - np.exp(-1.0 / (2.0 * sigma)))
    return (p**hw) * ((1.0 - p) ** (n - hw))


class MMD(nn.Module):
    """
    MMD² loss in Fourier space with gradient flow.
    Uses bitstring probabilities directly instead of discrete samples.


    TODO ⚠⚠ Right now it is computing full WHT transform of the two distributions. 
    Needs to be changed such that, the target transform is passed from outside, 
    and the model probability distribution if computed only the PARTIAL wht,
    that needs to be differentiable, to be implemented⚠⚠
    """
    
    def __init__(self, N: int, target_dist: np.ndarray, mmd_sigma: float = 1.0):
        """
        Args:
            N: Bitstring length
            target_dist: Target probability distribution (2^N,)
            mmd_sigma: Kernel bandwidth
        """
        super().__init__()
        self.N = N
        self.mmd_sigma = mmd_sigma
        
        # Precompute target distribution, keeping the process out of differentiation
        self.register_buffer(
            'target_dist',
            torch.from_numpy(target_dist / target_dist.sum()).float()
        )
        
        # Precompute MMD kernel weights in Fourier space
        # TODO: like this cannot scale, cause we are looping on 2**N
        weights = np.array(
            [mmd_kernel_weight(k.bit_count(), N, mmd_sigma) for k in range(2**N)],
            dtype=np.float32
        )
        self.register_buffer('kernel_weights', torch.from_numpy(weights))
    
    def forward(self, samples: torch.Tensor) -> torch.Tensor:
        """
        Compute differentiable MMD² loss.
        
        Args:
            samples: (batch_size, N) bitstrings from the model
        
        Returns:
            loss: scalar torch tensor (differentiable)
        """
        batch_size = samples.size(0)
        device = samples.device
        
        # BRIDGE: Construct the 2^N empirical distribution from N-dim samples.
        # This keeps the gradients flowing back to the N bits.
        indices = torch.arange(2**self.N, device=device)
        
        # Create a (2^N, N) matrix of all possible ideal bitstrings
        bit_matrix = ((indices.unsqueeze(1) >> torch.arange(self.N, device=device)) & 1).float()
        
        # Probability of each sample matching each 2^N state
        # samples: (batch_size, 1, N), bit_matrix: (1, 2^N, N)
        match_probs = samples.unsqueeze(1) * bit_matrix.unsqueeze(0) + \
                      (1 - samples.unsqueeze(1)) * (1 - bit_matrix.unsqueeze(0))
                      
        # Joint probability over N bits -> (batch_size, 2^N)
        state_probs = match_probs.prod(dim=2)
        
        # Mean probability across batch (empirical distribution)
        empirical_dist = state_probs.mean(dim=0)  # (2^N,)
        
        # Transform both to Fourier space
        # TODO: we use full Fourier transform, differentiable
        # in the final one, fourier_target will be passed from outside
        fourier_emp = self._full_wht(empirical_dist)
        fourier_target = self._full_wht(self.target_dist)
        
        # MMD²
        diff = fourier_emp - fourier_target
        mmd_sq = torch.sum(self.kernel_weights * diff ** 2)
        
        return mmd_sq
        
    def _full_wht(self, p: torch.Tensor) -> torch.Tensor:
        """
        Differentiable approximation of WHT.
        For small N (≤12), we can compute exactly in PyTorch.
        """
        # For small N, construct Walsh-Hadamard matrix
        device = p.device
        N = self.N
        
        # Build Hadamard matrix recursively
        H = self._hadamard_matrix(N, device)
        
        # Transform: multiply by unnormalized Hadamard matrix
        return torch.matmul(H, p)
    
    def _hadamard_matrix(self, n: int, device: torch.device) -> torch.Tensor:
        """Construct unnormalized Hadamard matrix of size 2^n × 2^n."""
        if n == 0:
            return torch.tensor([[1.0]], device=device)
        
        H_prev = self._hadamard_matrix(n - 1, device)
        # Kronecker product: H_n = H_1 ⊗ H_{n-1}
        # H_1 = [[1, 1], [1, -1]]
        size_prev = 2 ** (n - 1)
        H = torch.zeros(2**n, 2**n, device=device)
        
        # Manual Kronecker product
        H[0:size_prev, 0:size_prev] = H_prev
        H[0:size_prev, size_prev:2**n] = H_prev
        H[size_prev:2**n, 0:size_prev] = H_prev
        H[size_prev:2**n, size_prev:2**n] = -H_prev
        
        return H

class Model(nn.Module):
    """
    Shallow single hidden layer model.
    """
    
    def __init__(self, N: int, k: float = 2.0, r: float = 1.0):
        """
        Args:
            N: Bitstring length
            k: Exponent for parameter scaling
            r: Fraction parameter
        """
        super().__init__()
        self.N = N
        self.k = k
        self.r = r
        
        # Calculate hidden dimension
        # Adjusted target params calculation since output is now N, not 2^N
        target_params = int((N ** k) * r)
        H = max(1, (target_params - N) // (N + N + 1))
        self.hidden_dim = H
        
        # Network outputs logits for each independent bit
        self.net = nn.Sequential(
            nn.Linear(N, H),
            nn.ReLU(),
            nn.Linear(H, N)  # CHANGED: Logit for each of the N bits
        )
    
    def forward(self, z: torch.Tensor, hard: bool = True) -> torch.Tensor:
        """
        Forward pass with Straight-Through Estimator (STE) for discrete sampling.
        
        Args:
            z: (batch_size, N) Gaussian noise
            hard: If True, return discrete bitstrings with gradients attached
        
        Returns:
            samples: (batch_size, N)
        """
        logits = self.net(z)
        probs = torch.sigmoid(logits)
        
        if hard:
            # Generate discrete samples (0 or 1)
            samples = (probs > 0.5).float()
            # STE: Pass gradients through the non-differentiable thresholding
            samples = samples.detach() - probs.detach() + probs
            return samples
            
        return probs
            
    def get_param_count(self) -> int:
        """Return total trainable parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def get_config(self) -> dict:
        """Return model configuration."""
        return {
            "N": self.N,
            "hidden_dim": self.hidden_dim,
            "k": self.k,
            "r": self.r,
            "target_params": int(self.N**self.k * self.r),
            "actual_params": self.get_param_count(),
        }

class Trainer:
    """Trainer with fully differentiable MMD² loss."""
    
    def __init__(
        self,
        model: Model,
        target_dist: np.ndarray,
        device: str = "cpu",
        mmd_sigma: float = 1.0,
        lr: float = 1e-3,
    ):
        """
        Args:
            model: Model instance
            target_dist: Target probability distribution (2^N,)
            device: "cpu" or "cuda"
            mmd_sigma: MMD kernel bandwidth
            lr: Learning rate
        """
        self.model = model.to(device)
        self.device = device
        self.N = model.N
        
        # Loss function
        self.mmd_loss = MMD(model.N, target_dist, mmd_sigma).to(device)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # Updating learning rate with Cosine Annealing:
        # https://docs.pytorch.org/docs/2.12/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html , based on:
        # https://arxiv.org/abs/1608.03983
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=200
        )
        
        self.history = {"loss": [], "lr": []}
    
    def train_step(self, batch_size: int = 128) -> float:
        """
        Single training step with full gradient flow.
        
        Args:
            batch_size: Number of samples per step
        
        Returns:
            Loss value
        """
        # Reset gradients to None
        self.optimizer.zero_grad()
        
        # Generate noise in input to the neural network
        z = torch.randn(batch_size, self.N, device=self.device)
        
        # Forward: get samples IMPORTART hard=False, or we get gradient vanishing from too rigid parameters.
        samples = self.model(z, hard=False)
        
        # Compute MMD²
        loss = self.mmd_loss(samples)
        
        # Backward (gradients clipping)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        self.history["loss"].append(loss.item())
        self.history["lr"].append(self.optimizer.param_groups[0]["lr"])
        
        return loss.item()
    
    def train(
        self,
        num_epochs: int = 100,
        batch_size: int = 128,
        eval_interval: int = 10,
    ):
        """
        Full training loop.
        
        Args:
            num_epochs: Number of epochs
            batch_size: Batch size
            eval_interval: Print interval
        """
        print("Starting training...")
        print(f"{'Epoch':<8} {'Loss':<12}")
        print("-" * 22)
        
        for epoch in range(num_epochs):
            loss = self.train_step(batch_size=batch_size)
            
            if (epoch + 1) % 10 == 0:
                self.scheduler.step()
            
            if (epoch + 1) % eval_interval == 0:
                print(f"{epoch + 1:<8} {loss:.6f}")
        
        print("-" * 22)
        print("Training complete!")
    
    def sample(self, num_samples: int) -> np.ndarray:
        """Generate samples as numpy array."""
        self.model.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, self.N, device=self.device)
            logits = self.model(z)
            bitstrings = self.model.sample_from_logits(logits, hard=True)
        return bitstrings.cpu().numpy()
    
    def get_history(self) -> dict:
        """Return training history."""
        return self.history



if __name__ == "__main__":
    from qfso.distributions import discretized_normal_probability, plot_distributions

    N = 8
    k = 3.0
    r = 1
    epochs=200

    model = Model(N=N, k=k, r=r)

    config = model.get_config()
    print("=" * 60)
    print("Model Configuration")
    print("=" * 60)
    print(f"Bitstring length (N):        {config['N']}")
    print(f"Hidden dimension (H):        {config['hidden_dim']}")
    print(f"Parameter scaling (k, r):    k={config['k']}, r={config['r']}")
    print(f"Target parameters (N^k*r):   {config['target_params']}")
    print(f"Actual parameters:           {config['actual_params']}")
    print()
    
    # Target
    # In our test this should be the distribution of the quantum circuit we are trying to spoof
    # Of course if spoofing we do not have access to this, but just to the truncated wht transform.
    # We need to adapt the class MMD such that instead of doing the full WHT transform of this, takes
    # the pre-computed truncaded transform.
    target_dist = discretized_normal_probability((-10, 9), 2**N)

    
    # Create trainer
    trainer = Trainer(
        model,
        target_dist,
        mmd_sigma=1.0,
        lr=1e-1
    )
    
    print("=" * 60)
    print("Training")
    print("=" * 60)
    print()
    
    trainer.train(
        num_epochs=epochs,
        batch_size=128,
        eval_interval=10
    )
    
    # Evaluate
    # print()
    # print("=" * 60)
    # print("Sample Results")
    # print("=" * 60)
    
    # samples = trainer.sample(10)
    # print(f"\nGenerated {samples.shape[0]} samples:")
    # for i, s in enumerate(samples):
    #     binary_str = "".join(map(str, s.astype(int)))
    #     print(f"  {i}: {binary_str}")