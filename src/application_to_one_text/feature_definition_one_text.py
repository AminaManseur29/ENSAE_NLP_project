""" Contient des fonctions de prétraitement pour la définition des caractértistiques stylistiques,
caractéristiques de complexité et psychologiques. """

import pandas as pd
import numpy as np
import nltk
import string
import re
from nltk.corpus import stopwords, words
from nltk.corpus import sentiwordnet as swn
from nltk.tag import pos_tag

# Télécharge les ressources nécessaires pour NLTK
nltk.download('words')
nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('sentiwordnet')

stop_words = set(stopwords.words('english'))
english_vocab = set(words.words())

# -----------------------
# Extraction des caractéristiques stylistiques
# -----------------------


# Fonction pour nettoyer un mot
def clean_word(word):
    return ''.join(c for c in word if c.isalpha())


# Fonction pour effectuer le POS tagging sur une liste de tokens
def batch_pos_tag(tokenized_text):
    """Effectue le POS tagging pour tous les textes d'un coup."""
    return pos_tag(tokenized_text)


class extract_stylistic_features:
    """ Extrait les caractéristiques stylistiques du texte. """
    def __init__(self, texts, tokenized_texts):
        self.texts = texts
        self.tokenized_texts = tokenized_texts

    def quote_frequency(self, text):
        """ Calcule la fréquence des citations dans le texte. """
        quote_patterns = r'["\'«»]'
        # Trouver toutes les occurrences de ces guillemets dans le texte
        quotes = re.findall(quote_patterns, text)
        # Calculer la fréquence des guillemets
        quote_count = len(quotes)
        total_characters = len(text)
        # Calculer la fréquence en pourcentage par rapport au nombre total de caractères
        quote_frequency = quote_count / total_characters if total_characters > 0 else 0
        return quote_count, quote_frequency

    def punctuation_features(self, text):
        """ Calcule le nombre total de ponctuations
        et le nombre de ponctuations uniques dans le texte. """
        punctuation_count = sum(1 for char in text if char in string.punctuation)
        unique_punctuation_count = len(set(char for char in text if char in string.punctuation))
        return punctuation_count, unique_punctuation_count

    def exclamation_frequency(self, text):
        """ Calcule la fréquence des points d'exclamation dans le texte. """
        exclamation_count = text.count('!')
        return exclamation_count / len(text.split()) if len(text.split()) > 0 else 0

    def stopword_frequency(self, tokenized_text):
        """ Calcule la fréquence des mots vides dans le texte. """
        stopword_count = sum(1 for word in tokenized_text if word.lower() in stop_words)
        return stopword_count / len(tokenized_text) if len(tokenized_text) > 0 else 0

    def camel_case_frequency(self, tokenized_text):
        """ Calcule la fréquence des mots en camel case dans le texte. """
        camel_case_count = sum(
            1 for word in tokenized_text if word[0].isupper() and any(c.islower() for c in word[1:])
            )
        return camel_case_count / len(tokenized_text) if len(tokenized_text) > 0 else 0

    def negation_frequency(self, tokenized_text):
        """ Calcule la fréquence des mots de négation dans le texte. """
        negation_words = {"not", "no", "never", "none", "nobody", "nothing"}
        negation_count = sum(1 for word in tokenized_text if word.lower() in negation_words)
        return negation_count / len(tokenized_text) if len(tokenized_text) > 0 else 0

    def proper_nouns_frequency(self, tagged_text):
        """Calcule la fréquence des noms propres dans le texte."""
        proper_noun_count = sum(1 for word, tag in tagged_text if tag in ['NNP', 'NNPS'])
        return proper_noun_count / len(tagged_text) if len(tagged_text) > 0 else 0

    def user_mentions_frequency(self, text):
        """ Calcule la fréquence des mentions d'utilisateur dans le texte. """
        user_mentions = text.count('@')
        return user_mentions / len(text.split()) if len(text.split()) > 0 else 0

    def hashtags_frequency(self, text):
        """ Calcule la fréquence des hashtags dans le texte. """
        hashtags = text.count('#')
        return hashtags / len(text.split()) if len(text.split()) > 0 else 0

    def misspelled_words_frequency(self, tokenized_text):
        """Calcule la fréquence des mots mal orthographiés parmi les vrais mots."""
        # Nettoyer les mots et ne garder que ceux qui sont vraiment alphabétiques
        clean_tokens = [clean_word(word).lower() for word in tokenized_text if clean_word(word)]

        if len(clean_tokens) == 0:
            return 0  # éviter une division par zéro

        misspelled_count = sum(1 for word in clean_tokens if word not in english_vocab)
        return misspelled_count / len(clean_tokens)

    def oov_frequency(self, tokenized_text):
        """ Calcule la fréquence des mots hors vocabulaire dans le texte. """
        oov_count = sum(1 for word in tokenized_text if list(swn.senti_synsets(word)))
        return oov_count / len(tokenized_text) if len(tokenized_text) > 0 else 0

    def noun_frequency(self, tagged_words):
        noun_count = sum(1 for word, tag in tagged_words if tag in ['NN', 'NNS', 'NNP', 'NNPS'])
        return noun_count / len(tagged_words) if len(tagged_words) > 0 else 0

    def past_tense_frequency(self, tagged_words):
        past_tense_count = sum(1 for word, tag in tagged_words if tag in ['VBD', 'VBN'])
        return past_tense_count / len(tagged_words) if len(tagged_words) > 0 else 0

    def verb_frequency(self, tagged_words):
        verb_count = sum(1 for word, tag in tagged_words if tag.startswith('VB'))
        return verb_count / len(tagged_words) if len(tagged_words) > 0 else 0

    def interrogative_frequency(self, tagged_words):
        interrogative_count = sum(1 for word, tag in tagged_words if tag in ['WRB', 'WDT', 'WP'])
        return interrogative_count / len(tagged_words) if len(tagged_words) > 0 else 0

    def extract_features(self):
        """ Extrait toutes les caractéristiques stylistiques pour chaque texte. """
        text = self.texts
        tokenized_text = self.tokenized_texts
        tagged_text = batch_pos_tag(tokenized_text)

        ponctuation_count, unique_punctuation_count = self.punctuation_features(text)
        features = {
            "quote_frequency": self.quote_frequency(text),
            "punctuation_count": ponctuation_count,
            "unique_punctuation_count": unique_punctuation_count,
            "exclamation_frequency": self.exclamation_frequency(text),
            "stopword_frequency": self.stopword_frequency(tokenized_text),
            "camel_case_frequency": self.camel_case_frequency(tokenized_text),
            "negation_frequency": self.negation_frequency(tokenized_text),
            "proper_noun_frequency": self.proper_nouns_frequency(tagged_text),
            "user_mentions_frequency": self.user_mentions_frequency(text),
            "hashtag_frequency": self.hashtags_frequency(text),
            "misspelled_words": self.misspelled_words_frequency(tokenized_text),
            "oov_frequency": self.oov_frequency(tokenized_text),
            "noun_frequency": self.noun_frequency(tagged_text),
            "past_tense_frequency": self.past_tense_frequency(tagged_text),
            "verb_frequency": self.verb_frequency(tagged_text),
            "interrogative_frequency": self.interrogative_frequency(tagged_text),
        }

        return features

# -----------------------
# Extraction des caractéristiques de complexité
# -----------------------


class extract_complexity_features():
    def __init__(self, texts, tokenized_texts):
        self.texts = texts
        self.tokenized_texts = tokenized_texts

    def word_count_and_mean_length(self, tokenized_text):
        """ Calcule le nombre total de mots et la longueur moyenne des mots dans le texte. """
        word_count = len(tokenized_text)
        mean_length = sum(
            len(word) for word in tokenized_text
            ) / word_count if word_count > 0 else 0
        return word_count, mean_length

    def ttr(self, tokenized_text):
        """ Calcule le type-token ratio (TTR) du texte. """
        unique_words = set(tokenized_text)
        ttr_value = len(unique_words) / len(tokenized_text) if len(tokenized_text) > 0 else 0
        return ttr_value

    def mtld(self, tokenized_text, threshold=0.72):
        """ Calcule le Measure of Textual Lexical Diversity (MTLD) du texte. """
        n = len(tokenized_text)
        if n == 0:
            return 0
        mtld_value = 0
        current_ttr = 0
        current_count = 0
        for i in range(n):
            current_count += 1
            current_ttr = len(set(tokenized_text[:i+1])) / current_count
            if current_ttr < threshold:
                mtld_value += current_count / len(set(tokenized_text[:i+1]))
                current_count = 0
        return mtld_value

    def extract_features(self):
        """ Extrait les caractéristiques de complexité du texte. """
        tokenized_text = self.tokenized_texts
        features = {
            "word_count": self.word_count_and_mean_length(tokenized_text)[0],
            "mean_word_length": self.word_count_and_mean_length(tokenized_text)[1],
            "ttr": self.ttr(tokenized_text),
            "mtld": self.mtld(tokenized_text)
        }
        return features

# -----------------------
# Extraction des caractéristiques psychologiques
# -----------------------


class extract_psychological_features():
    def __init__(self, texts, tokenized_texts):
        self.texts = texts
        self.tokenized_texts = tokenized_texts

    def sentiment_score(self, tokenized_texts):
        """ Calcule le score de sentiment du texte. """
        sentiment_score = 0
        for word in tokenized_texts:
            synsets = list(swn.senti_synsets(word))
            if synsets:
                sentiment_score += synsets[0].pos_score() - synsets[0].neg_score()
        return sentiment_score / len(tokenized_texts) if len(tokenized_texts) > 0 else 0

    def extract_features(self):
        """ Extrait les caractéristiques psychologiques du texte. """
        tokenized_text = self.tokenized_texts
        features = {
            "sentiment_score": self.sentiment_score(tokenized_text)
        }
        return pd.DataFrame(features)


def compute_handcrafted_features_one_text(text):
    """Extrait toutes les caractéristiques stylistiques,
    de complexité et psychologiques d'un seul texte."""

    # 1. Tokenisation simple
    tokens = nltk.word_tokenize(text)

    # 2. Extraction des features
    stylistic_extractor = extract_stylistic_features(text, tokens)
    stylistic_features = stylistic_extractor.extract_features()

    complexity_extractor = extract_complexity_features(text, tokens)
    complexity_features = complexity_extractor.extract_features()

    psychological_extractor = extract_psychological_features(text, tokens)
    psychological_features = psychological_extractor.extract_features().to_dict(orient='records')[0]

    # 3. Fusion de toutes les features
    all_features = {**stylistic_features, **complexity_features, **psychological_features}

    # Transformer en numpy array (1, nb_features)
    feature_vector = np.array(list(all_features.values()), dtype=float).reshape(1, -1)

    return feature_vector
