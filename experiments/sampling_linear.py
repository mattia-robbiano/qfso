import pickle
import numpy as np
from qfso.models.approximate.probability import FactorizedDistribution, LinCombApproximation

def load_safe_model(filepath: str) -> LinCombApproximation:
    """
    Ricarica il dizionario di stato e ricostruisce gli oggetti completi.
    """
    with open(filepath, "rb") as f:
        state = pickle.load(f)
        
    components = []
    for comp_data in state["components"]:
        # Ricostruisce la singola distribuzione (le lambda verranno ricalcolate fresche dall'__init__)
        dist = FactorizedDistribution(
            independent_parities=comp_data["independent_parities"],
            probabilities=comp_data["probabilities"]
        )
        components.append(dist)
        
    # Ricostruisce e restituisce la combinazione lineare
    return LinCombApproximation(components, state["weights"])


def sample_from_lincomb(mixture, n_samples=1):
    """
    Campiona da una LinCombApproximation senza mai valutare l'intero spazio 2^n.
    """
    weights = np.array(mixture.weights)
    dists = mixture.probabilities
    
    assert np.all(weights >= 0), "I pesi devono essere positivi per campionare"
    weights = weights / np.sum(weights)
    
    # 1. Seleziona quale distribuzione usare per ogni sample
    chosen_indices = np.random.choice(len(weights), size=n_samples, p=weights)
    
    # 2. Campiona dalla singola FactorizedDistribution
    samples = []
    for idx in chosen_indices:
        dist = dists[idx]
        samples.append(dist.sample())
        
    return samples

if __name__ == "__main__":
    print("Caricamento modello...")
    
    # CORREZIONE 1: Utilizza la tua funzione custom per caricare correttamente l'oggetto
    model = load_safe_model("trained_combo_n484.pkl")
        
    n_qubits = model.probabilities[0].n
    print(f"Modello caricato con successo (n = {n_qubits})")
    
    n_tot = 100_000
    print(f"\nGenerazione di {n_tot} sample...")
    results = sample_from_lincomb(model, n_samples=n_tot)
    
    output_filename = "samples_combo.npy"
    np.save(output_filename, results)
    print(f"Tutti i sample salvati in: {output_filename}")
    
    print("\nVisualizzazione dei primi 5 sample (sanity check):")
    for i in range(5):
        s = results[i]
        print(f"Sample {i+1}: {s} (binary: {bin(s)})")