import pandas as pd
import re
import nltk

nltk.download('punkt')

# Function to clean text (remove unwanted characters, lowercasing, etc.)
def clean_text(text):
    text = text.lower()  # Convert text to lowercase
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text

def preprocess_data(df):
    # Apply the cleaning function to the 'Text' column
    df['cleaned_text'] = df['Text'].apply(clean_text)
    return df
