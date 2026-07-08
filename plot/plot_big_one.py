import pickle
import itertools
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configurazione stile pubblicazione (QTML-ready)
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 13,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 15,
    'pdf.fonttype': 42
})

if __name__ == "__main__":
    N = 484
    
    print("1. Caricamento dati...")
    with open('expvals_IqpSimulator_dwave.pkl', 'rb') as f:
        iqp_data = pickle.load(f)
    target_wht_k12 = iqp_data['expvals'] 
    
    with open('generated_spoofing_samples_v3.pkl', 'rb') as f:
        nn_samples = pickle.load(f)
    
    nn_spins = 1.0 - 2.0 * nn_samples
    num_samples = nn_spins.shape[0]

    # =================================================================
    # CALCOLO COEFFICIENTI (ORDINATI E SUB-SAMPLED)
    # =================================================================
    print("2. Elaborazione frequenze ordinate...")
    ops_k1 = list(itertools.combinations(range(N), 1))
    ops_k2 = list(itertools.combinations(range(N), 2))
    
    # K=1 (Tutte le 484 frequenze locali)
    nn_wht_k1 = nn_spins.mean(axis=0)
    target_plot_k1 = target_wht_k12[:len(ops_k1)]
    
    # K=2 (Sotto-campionamento per non affollare il grafico)
    stride = 50  
    sampled_ops_k2 = ops_k2[::stride]
    nn_wht_k2 = []
    for idxs in sampled_ops_k2:
        parity = nn_spins[:, idxs[0]] * nn_spins[:, idxs[1]]
        nn_wht_k2.append(parity.mean())
    nn_wht_k2 = np.array(nn_wht_k2)
    target_plot_k2 = target_wht_k12[len(ops_k1):][::stride]

    # K=3 (Generazione Horizon stocastico - Non visto in training)
    print("3. Estrazione stocastica ordine k=3...")
    num_k3_tests = 1000
    np.random.seed(42)
    nn_wht_k3 = []
    target_wht_k3 = []
    
    for _ in range(num_k3_tests):
        idxs = np.random.choice(N, 3, replace=False)
        parity_nn = nn_spins[:, idxs[0]] * nn_spins[:, idxs[1]] * nn_spins[:, idxs[2]]
        nn_wht_k3.append(parity_nn.mean())
        mean_field_target = target_wht_k12[idxs[0]] * target_wht_k12[idxs[1]] * target_wht_k12[idxs[2]]
        target_wht_k3.append(mean_field_target)

    # =================================================================
    # GENERAZIONE GRAFICO: 3 CANVAS SEPARATI
    # =================================================================
    print("4. Generazione dei canvas separati...")
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5.5))
    
    # Identifichiamo i limiti globali per normalizzare le diagonali y=x
    all_vals_seen = np.concatenate([target_plot_k1, nn_wht_k1, target_plot_k2, nn_wht_k2])
    lims_seen = [all_vals_seen.min() - 0.02, all_vals_seen.max() + 0.02]
    
    lims_k3_x = [min(target_wht_k3) - 0.02, max(target_wht_k3) + 0.02]
    lims_k3_y = [min(nn_wht_k3) - 0.02, max(nn_wht_k3) + 0.02]

    # --- CANVAS 1: 1-local (k=1) ---
    ax1.scatter(target_plot_k1, nn_wht_k1, color='#1f77b4', alpha=0.6, edgecolors='none', label='1-local frequencies')
    ax1.plot(lims_seen, lims_seen, 'k--', alpha=0.7, label='Perfect Match (y=x)')
    ax1.set_title("1-Local Spectrum (k=1)\n[Trained]")
    ax1.set_xlabel(r"Target IQP Coefficients $\langle \chi_1 \rangle$")
    ax1.set_ylabel(r"Neural Network Samples $\langle \chi_1 \rangle_{NN}$")
    ax1.set_xlim(lims_seen)
    ax1.set_ylim(lims_seen)
    ax1.legend(loc="upper left")

    # --- CANVAS 2: 2-local (k=2) ---
    ax2.scatter(target_plot_k2, nn_wht_k2, color='#ff7f0e', alpha=0.4, edgecolors='none', label='2-local parities')
    ax2.plot(lims_seen, lims_seen, 'k--', alpha=0.7, label='Perfect Match (y=x)')
    ax2.set_title("2-Local Spectrum (k=2)\n[Trained]")
    ax2.set_xlabel(r"Target IQP Coefficients $\langle \chi_2 \rangle$")
    ax2.set_ylabel(r"Neural Network Samples $\langle \chi_2 \rangle_{NN}$")
    ax2.set_xlim(lims_seen)
    ax2.set_ylim(lims_seen)
    ax2.legend(loc="upper left")

    # --- CANVAS 3: 3-local (k=3) 
    ax3.scatter(target_wht_k3, nn_wht_k3, color='#2ca02c', alpha=0.5, edgecolors='none', label='3-local (Unseen)')
    # Mostriamo la diagonale ideale per evidenziare il collasso verticale
    diag_lims = [min(lims_k3_x[0], lims_k3_y[0]), max(lims_k3_x[1], lims_k3_y[1])]
    ax3.plot(diag_lims, diag_lims, 'k--', alpha=0.7, label='Perfect Match (y=x)')
    ax3.set_title("3-Local Spectrum (k=3) [not trained]")
    ax3.set_xlabel(r"Reconstructed IQP Order-3 $\langle \chi_3 \rangle$")
    ax3.set_ylabel(r"Neural Network Predictions $\langle \chi_3 \rangle_{NN}$")
    ax3.set_xlim(lims_k3_x)
    ax3.set_ylim(lims_k3_y)
    ax3.legend(loc="upper left")

    # Titolo globale richiesto
    plt.suptitle("Spoofing IQP in MMD with shallow neural network \n 484 qubits D-WAVE dataset - MMD2 = 9.492e-04", y=0.98)
    plt.tight_layout()

    output_pdf = "qtml_canvas_separati_v3.pdf"
    plt.savefig(output_pdf, bbox_inches='tight', dpi=300)
    print(f"Fatto! Grafico a 3 canvas salvato in: {output_pdf}")
    plt.show()