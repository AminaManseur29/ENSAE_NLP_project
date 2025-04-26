# ENSAE_NLP_project

Utiliser un GPU pour calculer les embeddings et entraîner les modèles

Créer un environnement virtuel .venv avec uv, l'activer et installer les requirements :

pip install uv

uv venv .venv
source .venv/bin/activate

uv sync

python -m ipykernel install --user --name ensae-nlp --display-name "ENSAE NLP (uv)"

ctrl + shift + P pour reload window

puis ouvrir notebook et sélectionner le kernel qui vient d'être créé


Créer un fichier .env et y entrer son token openAI (ou huggingface en fonction des modèles utilisés)