""" Contient des fonctions qui permettent de calculer les embeddings d'un texte
avec 3 modèles choisis : BERT, RoBERTa et SentenceBERT. """

from transformers import AutoTokenizer, AutoModel
import torch
from sentence_transformers import SentenceTransformer

# Détection du device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------------
# Modèle BERT
# -------------------------------


model_name = "bert-base-cased"
tokenizer_bert = AutoTokenizer.from_pretrained(model_name)
model_bert = AutoModel.from_pretrained(model_name).to(device)
model_bert.eval()


def compute_bert_embedding(text, max_length=512):
    """Embedding d'un seul texte avec BERT."""
    inputs = tokenizer_bert(
        text, return_tensors="pt", padding=True, truncation=True, max_length=max_length
    )
    inputs = {key: val.to(device) for key, val in inputs.items()}

    with torch.no_grad():
        outputs = model_bert(**inputs)
    # Moyenne sur les tokens
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    return embedding


# -------------------------------
# Modèle RoBERTa
# -------------------------------


roberta_model_name = "roberta-base"
tokenizer_roberta = AutoTokenizer.from_pretrained(roberta_model_name)
model_roberta = AutoModel.from_pretrained(roberta_model_name).to(device)
model_roberta.eval()


def compute_roberta_embedding(text, max_length=512):
    """Embedding d'un seul texte avec RoBERTa."""
    inputs = tokenizer_roberta(
        text, return_tensors="pt", padding=True, truncation=True, max_length=max_length
    )
    inputs = {key: val.to(device) for key, val in inputs.items()}

    with torch.no_grad():
        outputs = model_roberta(**inputs)

    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    return embedding


# -------------------------------
# Modèle SentenceBERT
# -------------------------------

model_sbert = SentenceTransformer("all-MiniLM-L6-v2")


def compute_sbert_embedding(text):
    """Embedding d'un seul texte avec SentenceBERT."""
    embedding = model_sbert.encode(text, convert_to_numpy=True)
    return embedding
