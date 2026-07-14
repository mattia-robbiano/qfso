import torch
import numpy as np
import pickle

# Copia qui la definizione della classe Model per permettere a PyTorch di ricostruirlo
# (In alternativa, importala se hai il file originale)
from neural_network2 import Model

if __name__ == "__main__":
    # 1. Configura gli stessi identici iperparametri usati sul cluster
    N = 484
    k = 2.0
    r = 1.5
    
    # Inizializza il modello (su Mac userà la CPU o MPS, va benissimo la CPU per l'inference)
    device = torch.device("cpu")
    model = Model(N=N).to(device)
    
    # 2. Carica i pesi salvati dal cluster
    # Usiamo weights_only=True per motivi di sicurezza ed evitare warning
    weights_path = "spoofing_model_weights_v3.pth"
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()  # Imposta il modello in modalità valutazione
    print(f"Pesi caricati con successo da {weights_path}!")

    # 3. Genera i campioni (Inference)
    num_samples = 100_000  # Puoi alzare questo numero quanto vuoi
    print(f"Generazione di {num_samples} campioni in corso...")
    
    with torch.no_grad():  # Disabilita l'autograd per risparmiare memoria e CPU
        # Genera il rumore gaussiano di input
        z = torch.randn(num_samples, N, device=device)
        # Ottieni i bitstring discreti finali (grazie allo STE)
        bitstrings_tensor = model(z, hard=True)
        
        # Converte in formato NumPy per le tue analisi successive
        samples_np = bitstrings_tensor.cpu().numpy()

    print(f"Campionamento completato! Forma della matrice: {samples_np.shape}")
    
    # 4. Salva i campioni generati sul tuo Mac
    output_path = "generated_spoofing_samples_v3.pkl"
    with open(output_path, "wb") as f:
        pickle.dump(samples_np, f)
    print(f"Campioni salvati in locale su: {output_path}")