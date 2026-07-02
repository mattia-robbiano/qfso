import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

class EmpiricalDistributionModel(nn.Module):
    """
    Modello generativo MLP che produce campioni discreti (bitstring) 
    tramite il Gumbel-Softmax trick, ottimizzato con MMD².
    """
    def __init__(self, N: int, k: float, r: float):
        """
        N : int
            Lunghezza delle bitstring target e dimensione del vettore di rumore in input.
        k : float
            Esponente per il calcolo del numero di parametri target (N^k * r).
        r : float
            Fattore di scala in [0, 1] per il numero di parametri target.
        """
        super().__init__()
        self.N = N
        
        # Calcolo di H basato sul fatto che l'ultimo strato mappa H -> N * 2 logit
        # P = H * (N + 1) + H * (2N) + 2N = H * (3N + 1) + 2N
        target_params = int((N ** k) * r)
        H = max(1, (target_params - 2 * N) // (3 * N + 1))
        
        self.net = nn.Sequential(
            nn.Linear(N, H),
            nn.ReLU(),
            nn.Linear(H, N * 2)  # 2 logit per ognuno degli N bit (classe 0 e classe 1)
        )
        
    def forward(self, z: torch.Tensor, tau: float = 1.0, hard: bool = False) -> torch.Tensor:
        """
        Input: 
            z: Rumore casuale di shape (batch_size, N)
            tau: Temperatura del Gumbel-Softmax (valori bassi = più vicini a bit discreti)
            hard: Se True, ritorna bitstring discrete (0.0 o 1.0) ma mantiene il gradiente differenziabile
        Output:
            shape (batch_size, N) contenente le bitstring generate
        """
        # Reshape dei logit a (batch_size, N, 2) per separare le 2 classi (0 e 1) di ogni bit
        logits = self.net(z).view(-1, self.N, 2)
        
        # Il Gumbel-Softmax campiona in modo differenziabile dalle distribuzioni categoriche
        y = F.gumbel_softmax(logits, tau=tau, hard=hard, dim=-1)
        
        # Estraiamo la probabilità/valore associato al bit '1' (l'indice 1 della terza dimensione)
        bitstrings = y[:, :, 1]
        
        return bitstrings

    def get_actual_parameter_count(self) -> int:
        """Ritorna il numero esatto di parametri addestrabili della rete."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _gaussian_kernel(self, x: torch.Tensor, y: torch.Tensor, sigma: float) -> torch.Tensor:
        x_size, y_size, dim = x.size(0), y.size(0), x.size(1)
        x_expanded = x.unsqueeze(1).expand(x_size, y_size, dim)
        y_expanded = y.unsqueeze(0).expand(x_size, y_size, dim)
        distances = torch.pow(x_expanded - y_expanded, 2).sum(2)
        return torch.exp(-distances / (2 * sigma ** 2))

    def _mmd_squared_loss(self, x: torch.Tensor, y: torch.Tensor, sigma: float) -> torch.Tensor:
        k_xx = self._gaussian_kernel(x, x, sigma)
        k_yy = self._gaussian_kernel(y, y, sigma)
        k_xy = self._gaussian_kernel(x, y, sigma)
        return k_xx.mean() + k_yy.mean() - 2 * k_xy.mean()

    def fit(
        self, 
        X_train: torch.Tensor, 
        epochs: int = 100, 
        batch_size: int = 32, 
        lr: float = 0.005, 
        sigma_mmd: float = 0.1,
        initial_tau: float = 1.0,
        annealing_rate: float = 0.995
    ):
        """
        Avvia il loop di addestramento sul training set passato.

        Parameters:
        -----------
        X_train : torch.Tensor
            SHAPE RICHIESTA: (num_samples, N) -> Dataset di bitstring reali (vettori float).
        """
        if len(X_train.shape) != 2 or X_train.size(1) != self.N:
            raise ValueError(f"X_train deve avere shape (num_samples, {self.N}). Ricevuta: {X_train.shape}")

        optimizer = optim.Adam(self.parameters(), lr=lr)
        tau = initial_tau
        
        print(f"Parametri target calcolati: {int((self.N**k)*r)}")
        print(f"Parametri effettivi del modello: {self.get_actual_parameter_count()}")
        print("Inizio del training generativo discreto...")
        
        for epoch in range(epochs):
            self.train()
            epoch_loss = 0.0
            num_batches = 0
            
            permutation = torch.randperm(X_train.size(0))
            
            for i in range(0, X_train.size(0), batch_size):
                indices = permutation[i:i + batch_size]
                batch_real = X_train[indices]
                
                current_batch_size = batch_real.size(0)
                if current_batch_size < 2:
                    continue
                    
                optimizer.zero_grad()
                
                # Generazione rumore di input
                noise = torch.randn(current_batch_size, self.N)
                
                # Generazione di bitstring discrete (hard=True) ma differenziabili
                batch_generated = self(noise, tau=tau, hard=True)
                
                # Calcolo Loss MMD²
                loss = self._mmd_squared_loss(batch_real, batch_generated, sigma=sigma_mmd)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            # Riduzione graduale della temperatura del Gumbel-Softmax
            tau = max(0.5, tau * annealing_rate)
                
            if (epoch + 1) % 50 == 0 or epoch == 0:
                avg_loss = epoch_loss / num_batches
                print(f"Epoca [{epoch+1}/{epochs}] - MMD² Loss: {avg_loss:.6f} | Tau: {tau:.3f}")
                
        print("Training completato!")

    def sample(self, num_samples: int) -> torch.Tensor:
        """Metodo per generare bitstring pure discrete a fine training."""
        self.eval()
        with torch.no_grad():
            noise = torch.randn(num_samples, self.N)
            return self(noise, tau=0.5, hard=True)


if __name__ == "__main__":
    from qfso.distributions import discretized_normal_probability, plot_distributions

    # 1. Configurazione Iperparametri
    N = 10
    k = 3
    r = 1
    epochs = 1000
    batch_size = 1000
    lr = 0.01
    sigma_mmd = 0.1 * N

    # 2. Inizializzazione Modello
    model = EmpiricalDistributionModel(N=N, k=k, r=r)

    # 3. Generazione del Dataset Empirico (Training Set)
    num_samples = 1000
    p = discretized_normal_probability((-10, 9), 2**N)

    # Estrazione degli indici basata sulla distribuzione di probabilità p
    sampled_indices = np.random.choice(len(p), size=num_samples, p=p)
    
    # Conversione degli indici interi in bitstring binarie di lunghezza N
    X_train_np = (sampled_indices[:, None] >> np.arange(N)) & 1
    X_train = torch.tensor(X_train_np, dtype=torch.float32)

    # 4. Esecuzione del Fit
    model.fit(
        X_train, 
        epochs=epochs, 
        batch_size=batch_size, 
        lr=lr, 
        sigma_mmd=sigma_mmd
    )

    # 5. Test di generazione post-training: sample, calcolo della distribuzione empirica
    generated = model.sample(10000).cpu().numpy()
    generated_bits = (generated > 0.5).astype(np.int32)
    generated_indices = (generated_bits * (1 << np.arange(N))).sum(axis=1)
    counts = np.bincount(generated_indices, minlength=2**N)
    q_empirical = counts / counts.sum()
    plot_distributions(
        [q_empirical],
        labels=["distribution"],
        title=f"title",
    )