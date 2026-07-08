import pickle
import numpy as np

def sample_from_lincomb(mixture, n_samples=1):
    """
    Campiona da una LinCombApproximation senza mai valutare l'intero spazio 2^n.
    """
    # Estrae pesi e distribuzioni fattorizzate componenti
    weights = np.array(mixture.weights)
    dists = mixture.probabilities
    
    # Assicura che i pesi siano probabilità valide per il sampling
    assert np.all(weights >= 0), "I pesi devono essere positivi per campionare"
    weights = weights / np.sum(weights)
    
    # 1. Seleziona quale distribuzione usare per ogni sample
    chosen_indices = np.random.choice(len(weights), size=n_samples, p=weights)
    
    # 2. Campiona dalla singola FactorizedDistribution (O(n))
    samples = []
    for idx in chosen_indices:
        dist = dists[idx]
        samples.append(dist.sample())
        
    return samples

if __name__ == "__main__":
    print("Caricamento modello...")
    with open("trained_combo.pkl", "rb") as f:
        model = pickle.load(f)
        
    # Verifica che il numero di qubit (n) sia caricato correttamente
    # model.probabilities è la lista delle FactorizedDistribution
    n_qubits = model.probabilities[0].n
    print(f"Modello caricato (n = {n_qubits})")
    
    print("\nGenerazione di 10 sample...")
    results = sample_from_lincomb(model, n_samples=10)
    
    for i, s in enumerate(results):
        print(f"Sample {i+1}: {s} (binary: {bin(s)})")