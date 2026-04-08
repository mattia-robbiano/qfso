## Framework Teorico: La MMD nello Spazio di Walsh-Hadamard

La MMD esatta tra due distribuzioni $p$ e $q$ sui bitstrings può essere riscritta in funzione dei loro coefficienti di Fourier (o valori di aspettazione degli operatori di Pauli-Z diagonali):

$$MMD^2(p,q) = \sum_{\lambda \in \{0,1\}^n} w_\sigma(|\lambda|) \Delta_\lambda^2(p,q)$$

dove $\Delta_\lambda(p,q) := \hat{p}_\lambda - \hat{q}_\lambda$ è la differenza di segnale per l'operatore/frequenza $\lambda$.
L'aspetto fondamentale di questa formulazione è che il kernel gaussiano si traduce in una funzione di peso binomiale $w_\sigma(|\lambda|)$ parametrizzata dal peso di Hamming $|\lambda|$.

## Il Ruolo dei Parametri

* **$\sigma$ (Il "Soft Filter"):** $\sigma$ determina il parametro `p_MMD` per il campionamento della loss. 
    * **$\sigma$ elevato** (basso `p_MMD`): Sposta la massa del filtro verso i coefficienti a basso peso di Hamming. L'ottimizzazione vede solo le differenze marginali a basso ordine (correlazioni locali).
    * **$\sigma$ piccolo** (`p_MMD` $\approx 0.5$): Sposta il picco della distribuzione verso $|\lambda| \approx n/2$. Il filtro richiede al modello di matchare pattern di correlazione globale estremamente complessi.
* **`n_ops` (Varianza Monte Carlo):** Nel codice, non potendo sommare su tutti i $2^n$ operatori, campioniamo un sottoinsieme di `visible_ops` tramite `jax.random.binomial(..., p_MMD)`. `n_ops` **non** taglia le frequenze, ma controlla il numero di campioni MC per stimare la MMD in ogni step.
    * `n_ops` basso: gradienti stocastici (noisy), come un mini-batch piccolo in ML classico.
    * `n_ops` alto: stima accurata del gradiente MMD esatto.
* **`max_weight` (L' "Hard Cutoff" Architetturale):** Definisce i generatori fisici del circuito IQP (es. $e^{i \theta Z_i Z_j}$). È un limite espressivo invalicabile: se `max_weight = 2`, il modello non può generare nativamente segnali indipendenti ad alto peso di Hamming.
* **Entropia del Target (Lo Spettro):** Modulando la temperatura della distribuzione di Boltzmann *ground truth*, alteriamo lo spettro $\hat{p}_\lambda$:
    * Bassa Entropia (spiky): tante correlazioni  -> wht con picco verso destra (controllare in wht)
    * Alta Entropia (uniforme): poche correlazioni-> wht con picco vicino a 0   (controllare in wht)
* **`n_samples` (Bias Empirico / Shot Noise):** Calcolare $\Delta_\lambda$ usando un numero finito $s$ di campioni inietta rumore bianco su tutti i coefficienti. Empiricamente, si introduce un termine di bias asintotico scalabile come $\mathcal{O}(1/s)$, che "affoga" i segnali deboli ad alte frequenze.

---

### Esperimento A: Conflitto tra Espressività (`max_weight`) e Filtro ($\sigma$)
* **Setup:** Fissare un target a bassa entropia (molte correlazioni). Inizializzare un modello IQP con `max_weight = 2`.
* **Azione:** Addestrare due modelli: uno con $\sigma$ alto (filtro passa-basso) e uno con $\sigma$ basso (filtro passa-alto).
* **Ipotesi:** Il modello con $\sigma$ alto converge. Il modello con $\sigma$ basso fallisce o satura subito in minimi locali sub-ottimali, poiché il filtro richiede di ottimizzare frequenze che l'architettura non può fisicamente generare.

### Esperimento B: Scaling del Bias Empirico vs `n_samples`
* **Setup:** Calcolare unicamente il forward pass della MMD (nessun training) tra il dataset target e una sua ricampionatura dallo stesso modello teorico.
* **Azione:** Fissare `n_ops` molto alto (es. 10000) e $\sigma$. Variare `n_samples` logaritmicamente da $10$ a $10^5$.
* **Ipotesi:** Tracciando $\widehat{MMD}^2$ vs `n_samples` in scala log-log, emergerà una retta con pendenza $-1$, confermando che l'errore sistematico scala esattamente come $1/s$.

### Esperimento C: Lo Spettro dell'Entropia ($\sigma$ vs Entropia)
* **Setup:** Generare diversi dataset target variando l'entropia termodinamica. Usare un numero di samples $s$ adeguatamente grande.
* **Azione:** Calcolare la MMD tra questi target e un modello completamente random (distribuzione uniforme), effettuando uno sweep su $\sigma$.
* **Ipotesi:** Per target a bassa entropia, la MMD percepisce gradienti anche a $\sigma$ molto bassi. Per target ad alta entropia, la loss (il segnale utile) "muore" non appena $\sigma$ si sposta verso il basso.

### Esperimento D: Efficienza Stocastica (Ruolo di `n_ops`)
* **Setup:** Fissare $\sigma$, `max_weight` e `n_samples` in un regime stabile e convergente.
* **Azione:** Lanciare run di training indipendenti decrescendo drasticamente `n_ops` (es. 1000, 100, 10, 1). Tracciare la *Loss vs Epochs*.
* **Ipotesi:** Ridurre `n_ops` inietta rumore nell'ottimizzatore, ma l'aspettazione media dei gradienti è corretta (unbiased). Cerchiamo il valore critico di `n_ops` sotto il quale la varianza impedisce la convergenza, per massimizzare la velocità computazionale di ogni epoch.





### Il vero ruolo di `n_ops`: La stima Monte Carlo nello spazio di Walsh-Hadamard

L'equazione esatta della MMD in spazio WH è una somma su tutti i $2^n$ possibili operatori di Pauli-Z (o frequenze $\lambda$):
$$MMD^2(p,q) = \sum_{\lambda \in \{0,1\}^n} w_\sigma(|\lambda|) \Delta_\lambda^2(p,q)$$

Calcolare questa somma esatta per $n$ grande è intrattabile. Quello che il codice sta facendo con `jax.random.binomial` è **campionare $\lambda$ tramite metodo Monte Carlo** direttamente dalla distribuzione dei pesi $w_\sigma(|\lambda|)$! 
Il parametro `p_MMD` (che matematicamente è $p_\sigma$ in "note giulio.pdf") controlla la probabilità che ogni singolo qubit entri a far parte del generatore di Pauli. 

Quindi, la loss che stiamo effettivamente ottimizzando nel codice è una *stima non distorta* (unbiased estimator) della MMD:
$$\widehat{MMD}^2 \approx \frac{1}{n_{ops}} \sum_{k=1}^{n_{ops}} \Delta_{\lambda_k}^2(p,q)$$

Alla luce di questo, ecco l'architettura aggiornata dei nostri parametri e degli esperimenti che dovresti lanciare:

#### 1. $\sigma$ (ovvero `p_MMD`): Il Centro di Massa del Filtro
Come prima, $\sigma$ determina `p_MMD`. 
* Se $\sigma$ è grande, `p_MMD` è piccolo: campioneremo operatori (le `visible_ops`) molto sparsi, principalmente a peso 1 o 2.
* Se $\sigma$ è piccolo, `p_MMD` si avvicina a 0.5: le `visible_ops` avranno in media un peso di Hamming pari a $n/2$. Il modello sarà forzato a matchare le correlazioni globali di altissimo ordine.

#### 2. `n_ops`: La Risoluzione (Varianza) del Paesaggio di Loss
Poiché stiamo facendo un'integrazione Monte Carlo, `n_ops` non taglia le frequenze, ma **controlla la varianza della nostra loss**. 
* Se usi un `n_ops` molto basso (es. `n_ops=10`), stai calcolando la distanza tra le distribuzioni basandoti solo su 10 operatori casuali per step di ottimizzazione. Avrai gradienti estremamente stocastici (noisy gradients).
* Aumentando `n_ops`, la stima della MMD diventa più fedele al valore esatto. L'esperimento qui è: *qual è il valore critico di `n_ops` (in funzione dei qubit $n$) per cui la varianza del gradiente permette ancora la convergenza dell'optimizer?*

#### 3. `max_weight`: L'Hard Cutoff Architetturale (Espressività)
Come hai giustamente sottolineato, l'architettura del circuito è definita da `max_weight`. Questo è il limite "fisico" del modello. Se `max_weight = 2`, il circuito IQP applicherà solo gate $e^{i \theta Z_i Z_j}$. 

#### 4. `n_samples`: Il Bias Empirico (Shot Noise dei Dati)
I samples estratti per formare il dataset (`ground_truth`) entrano nel calcolo di $\Delta_{\lambda_k}(p,q) = \langle Z_{\lambda_k} \rangle_p - \langle Z_{\lambda_k} \rangle_q$. Avere un numero di samples $s$ finito introduce quel bias $\mathcal{O}(1/s)$ di cui parla Giulio.

---

### La Pipeline degli Esperimenti (Aggiornata)

Per supportare il "note giulio.pdf" e il tuo studio sullo spettro, ecco i tre esperimenti cruciali che puoi impostare adesso:

**Esperimento A: Conflitto tra Espressività (`max_weight`) e Filtro ($\sigma$)**
1. Fissa un target $p$ ad alta complessità (entropia media/bassa, forti correlazioni).
2. Costruisci il modello IQP con `max_weight = 2` (solo correlazioni locali).
3. Addestra il modello due volte:
   * **Run 1:** Usa $\sigma$ alto (`p_MMD` piccolo). Il filtro richiede di matchare operatori a basso peso. Il modello dovrebbe convergere bene, perché la sua architettura (`max_weight=2`) copre quelle frequenze.
   * **Run 2:** Usa $\sigma$ piccolo (`p_MMD` vicino a 0.5). La loss compilerà un batch di `visible_ops` con pesi alti. Il modello non ha la capacità architetturale di matchare quelle frequenze. **Ipotesi da verificare:** L'addestramento fallisce o la loss satura subito a un valore sub-ottimale.

**Esperimento B: Scaling della MMD Empirica vs `n_samples`**
Verifichiamo numericamente il bias teorico dell'equazione (9) di Giulio.
1. Valuta la MMD (senza addestramento, solo un *forward pass* del blocco di codice che mi hai incollato) tra il dataset target e una sua ricampionatura.
2. Mantieni fissi `n_ops` (molto alto, per azzerare la varianza MC sugli operatori) e $\sigma$.
3. Fai variare `n_samples` logaritmicamente (da 10 a $10^5$). 
4. Plottando $\widehat{MMD}^2$ vs `n_samples` in scala log-log, dovresti ottenere una retta con pendenza $-1$, confermando che l'errore scala come $1/s$ a causa del "rumore bianco" del campionamento su tutti gli operatori.

**Esperimento C: Efficienza Stocastica (Ruolo di `n_ops`)**
Qui vogliamo vedere quanto è tollerante il framework:
1. Fissa $\sigma$, `max_weight` e `n_samples` in un regime dove sai che il modello converge bene.
2. Lancia training separati riducendo `n_ops` (es. 1000, 100, 10, 1).
3. Vuoi tracciare la curva di addestramento (Loss vs Epochs). L'ipotesi è che, proprio come nel *Stochastic Gradient Descent* in ML classico, un `n_ops` ridotto (mini-batch in spazio WH) introduce rumore ma potrebbe comunque navigare la landscape media, possibilmente in meno tempo di calcolo.


Nel nostro contesto, lo spettro di Walsh-Hadamard dei bitstrings coincide con i valori di aspettazione degli operatori diagonali di Pauli-Z. Se indichiamo con $\lambda \in \{0,1\}^n$ la frequenza, il suo peso di Hamming $|\lambda|$ rappresenta l'ordine della correlazione (1-corpo, 2-corpi, ecc.). ### Il ruolo di $\sigma$: Come mostrato nell'equazione (1) (nostro overleaf) la MMD al quadrato può essere scritta in modo esatto nello spazio WH come: $$MMD^2(p,q) = \sum_{\lambda} w_\sigma(|\lambda|) \Delta_\lambda^2(p,q)$$ dove $\Delta_\lambda(p,q) := \hat{p}_\lambda - \hat{q}_\lambda$ è la differenza tra i coefficienti di Fourier. Il parametro del kernel gaussiano $\sigma$ definisce la funzione di peso $w_\sigma(|\lambda|)$, che ha una forma binomiale. Questo è il tuo **filtro**: * **$\sigma$ elevato**: Sposta la massa del filtro verso i coefficienti a basso peso di Hamming, ignorando le alte frequenze. L'ottimizzazione tramite MMD vedrà solo le differenze marginali a basso ordine. * **$\sigma$ piccolo**: Sposta il picco della distribuzione binomiale verso $|\lambda| \approx n/2$. Il filtro diventa un "passa-alto" o passa-banda centrato a metà spettro, richiedendo al modello di matchare pattern di correlazione estremamente complessi. ### Il ruolo di max_weight: L'Hard Cutoff Architetturale Mentre $\sigma$ pesa in fase di *valutazione* (loss function), max_weight (e il conseguente rank massimo dei generatori scelti, ad esempio tramite gates_from_covariance in iqpopt) impone un **taglio netto (hard cutoff)** in fase di *generazione*. Se il tuo circuito IQP ha solo gate generati da pesi $\le 2$, il modello $q$ non ha la capacità di generare nativamente segnali indipendenti ad alto peso $|\lambda|$. La sua "banda di trasmissione" è architetturalmente limitata. L'esperimento interessante qui è vedere il *mismatch*: cosa succede se max_weight è limitato a pesi $\le 2$, ma usi un $\sigma$ molto piccolo che impone alla mmd_loss_samples di penalizzare le differenze sui pesi alti? L'ottimizzazione fallirà o si assesterà in minimi locali sub-ottimali. ### 4. Il ruolo dei campioni ($s$): Il Bias di Stima (Shot Noise) Quando calcoliamo la MMD in via empirica usando la mmd_loss_samples, non abbiamo $p$, ma usiamo $s$ campioni estratti da $p$. L'aspettazione della MMD empirica introduce un termine di bias dipendente dai campioni $s$. Come si evince dal riarrangiamento delle equazioni nel testo (seppur non sia esplosa l'eq. 9 per intero), emerge un termine asintotico proporzionale a $\frac{1}{s}$. In spazio WH, il campionamento finito inietta "rumore bianco" (shot noise) su tutti i coefficienti. Se il tuo segnale vero $\Delta_\lambda$ sui pesi alti è piccolo, verrà interamente affogato dalla varianza $\sim 1/s$. ### Come impostare l'esperimento numerico: Per validare questa teoria, ti suggerisco questa pipeline da far girare con iqpopt: 1. **Test del Bias Finita (Effetto $s$):** Fissa un modello target $p$ a media entropia. Usa la mmd_loss_samples tra $p$ empirico e $p$ stesso varindo i samples $s$ in range $[100, 100000]$. Mostra in un plot log-log che la loss scala asintoticamente come $\mathcal{O}(1/s)$, confermando le stime di "note giulio.pdf". 2. **Scanner dello Spettro ($\sigma$ vs Entropia):** Genera target a diverse entropie. Fissa un numero di samples $s$ adeguatamente grande. Ora usa mmd_loss_samples misurando la distanza dal target a un modello completamentre random (distribuzione uniforme). Fai uno sweep di $\sigma$. Dovresti vedere che la MMD è sensibile al target a bassa entropia anche a $\sigma$ più bassi, mentre per i target ad alta entropia il gradiente "muore" se $\sigma$ esce dai bound della sigma_heuristic. 3. **Impatto di max_weight (Hard vs Soft cutoff):** Addestra il tuo circuito IQP limitato a weight-2 variando $\sigma$. Dimostra che quando $\sigma$ è piccolo (il filtro della MMD chiede alte frequenze), il modello non riesce ad abbassare la loss perché limitato dall'assenza di generatori complessi.