# embedding_generation.py
import pandas as pd
import re
from nltk.tokenize import word_tokenize
import nltk
import torch
from transformers import BertTokenizer, BertModel
import numpy as np

# Download nltk punkt for tokenization
nltk.download('punkt')


# Function to clean text (remove unwanted characters, lowercasing, etc.)
def clean_text(text):
    """ Preprocess the text by lowering case, removing unwanted characters and extra spaces. """
    text = text.lower()  # Convert text to lowercase
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text


# Function to generate BERT embeddings for the cleaned text
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')


def generate_embeddings(text):
    """ This function generates BERT embeddings for a given text. """
    # Clean the text first
    text = clean_text(text)

    # Handle empty or null values
    if not isinstance(text, str) or text.strip() == "":
        text = "[EMPTY]"  # Placeholder for empty values

    # Tokenize and generate BERT embeddings
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = bert_model(**inputs)

    # Return the embeddings of the [CLS] token
    return outputs.last_hidden_state[:, 0, :].squeeze().numpy()

