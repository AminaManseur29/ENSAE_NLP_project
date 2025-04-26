from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


# -------------------------------
# Modèle BERT
# -------------------------------

# Initialisation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "bert-base-cased"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)
model.eval()


# Fonction batchée pour obtenir les embeddings
def compute_bert_embeddings(texts, batch_size=32, max_length=512):
    all_embeddings = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Calcul des embeddings BERT"):
        batch_texts = texts[i:i+batch_size]

        # Tokenisation batch
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True,
                           truncation=True, max_length=max_length)

        inputs = {key: val.to(device) for key, val in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        # Moyenne sur la séquence pour chaque exemple
        batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        all_embeddings.append(batch_embeddings)

    return np.vstack(all_embeddings)


# -------------------------------
# Modèle RoBERTa
# -------------------------------

# Chargement du modèle RoBERTa
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
roberta_model_name = "roberta-base"

roberta_tokenizer = AutoTokenizer.from_pretrained(roberta_model_name)
roberta_model = AutoModel.from_pretrained(roberta_model_name).to(device)
roberta_model.eval()


# Fonction d'embedding RoBERTa
def compute_roberta_embeddings(texts, batch_size=32, max_length=512, save_path=None):
    if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
        raise ValueError("L'entrée doit être une liste de chaînes de caractères (List[str])")

    all_embeddings = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Calcul des embeddings RoBERTa"):
        batch_texts = texts[i:i+batch_size]

        inputs = roberta_tokenizer(batch_texts,
                                   return_tensors="pt",
                                   padding=True,
                                   truncation=True,
                                   max_length=max_length)

        inputs = {key: val.to(device) for key, val in inputs.items()}

        with torch.no_grad():
            outputs = roberta_model(**inputs)

        # Moyenne sur les tokens
        batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        all_embeddings.append(batch_embeddings)

    all_embeddings = np.vstack(all_embeddings)

    if save_path:
        np.save(save_path, all_embeddings)

    return all_embeddings


# -------------------------------
# Modèle SentenceBERT
# -------------------------------

# Chargement du modèle SBERT
sbert_model = SentenceTransformer("all-MiniLM-L6-v2")


# Fonction d'embedding SBERT
def compute_sbert_embeddings(texts, batch_size=32, save_path=None):
    if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
        raise ValueError("L'entrée doit être une liste de chaînes de caractères (List[str])")

    embeddings = sbert_model.encode(texts,
                                    batch_size=batch_size,
                                    show_progress_bar=True,
                                    convert_to_numpy=True)

    if save_path:
        np.save(save_path, embeddings)

    return embeddings
