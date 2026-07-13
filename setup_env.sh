#!/bin/bash

# Esci immediatamente se un comando fallisce
set -e

echo "=== Pulizia sicura di ~/.local tramite rsync ==="
rsync -a --delete ~/empty/ ~/.local/lib/python3.11/site-packages/

echo "=== Caricamento moduli creazione environment ==="
module purge
module load Python/3.11.5-GCCcore-13.2.0
module load CUDA/12.1.0

python -m venv .venv
source .venv/bin/activate

echo "Configurazione variabili di sicurezza PIP ==="
export PIP_USER=false
export PYTHONNOUSERSITE=1

echo "=== Installazione qfso ==="
pip install -e . -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

echo "=== Verifica finale dell'ambiente ==="
python -c "import jax; print('JAX caricato correttamente su:', jax.devices())"
python -c "import qfso; print('Modulo qfso installato correttamente!')"

echo "=== Configurazione completata con successo! ==="