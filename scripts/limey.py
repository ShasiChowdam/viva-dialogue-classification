import numpy as np
import pandas as pd
import os
import joblib
import torch
import re
import nltk
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from transformers import BertTokenizer, BertModel
from lime.lime_tabular import LimeTabularExplainer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from matplotlib.patches import Patch

# Download NLTK tokenizer (if not already available)
nltk.download('punkt')

# Initialize BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# Ensure interactive mode is off
plt.ioff()

# Define paths
data_path = r"C:\Users\sasic\PycharmProjects\ML_project\data\Final Viva Text.csv"
embeddings_path = "embeddings.npy"
model_path = "models/svm_model_bert.pkl"
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

# Function to clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text

# Function to generate BERT embeddings
def generate_embeddings(texts):
    embeddings = []
    for text in texts:
        text = clean_text(text)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = bert_model(**inputs)
        embeddings.append(outputs.last_hidden_state[:, 0, :].squeeze().numpy())  # CLS token embedding
    return np.array(embeddings)

# Load dataset
df = pd.read_csv(data_path)
texts = df["Text"].tolist()  # Update with actual column name
labels = df["Label"].tolist()  # Update with actual column name

# Tokenize texts
tokenized_texts = [word_tokenize(text) for text in texts]

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)  # Convert to numeric

# Generate embeddings if not already saved
if not os.path.exists(embeddings_path):
    print("Generating embeddings...")
    X = generate_embeddings(texts)
    np.save(embeddings_path, X)
    print("Embeddings saved.")
else:
    print("Loading existing embeddings...")
    X = np.load(embeddings_path)

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and save model if not already present
if not os.path.exists(model_path):
    print("Training SVM model...")
    svm_model = SVC(probability=True, kernel='linear', random_state=42)
    svm_model.fit(X_train, y_train)
    joblib.dump(svm_model, model_path)
    print("Model trained and saved.")
else:
    print("Loading existing model...")
    svm_model = joblib.load(model_path)

# Function to apply LIME explanation
def apply_lime_explanation(X, model, tokenized_texts, indices_to_explain):
    """ Apply LIME explanation for selected indices with enhanced visualization. """

    explainer = LimeTabularExplainer(
        training_data=X,
        mode='classification',
        class_names=label_encoder.classes_,
        discretize_continuous=True
    )

    for i in indices_to_explain:
        if i >= len(X):
            print(f"Index {i} out of bounds. Skipping.")
            continue

        # Get LIME explanation for the instance
        exp = explainer.explain_instance(X[i], model.predict_proba, num_features=10)

        # Extract feature importance
        exp_map = exp.as_map()[1]
        feature_indices = [idx for idx, _ in exp_map]
        feature_values = [v for _, v in exp_map]

        # Retrieve actual words using tokenized_texts
        feature_names = [tokenized_texts[i][idx] if idx < len(tokenized_texts[i]) else f"Feature {idx}"
                         for idx in feature_indices]

        # Sort features by absolute importance
        sorted_indices = np.argsort(np.abs(feature_values))[::-1]
        feature_names = [feature_names[idx] for idx in sorted_indices]
        feature_values = [feature_values[idx] for idx in sorted_indices]

        # Define colors (green = question, red = not question)
        colors = ['green' if v > 0 else 'red' for v in feature_values]

        # Generate visualization
        plt.figure(figsize=(10, 6))
        bars = plt.barh(feature_names, feature_values, color=colors, edgecolor="black")

        # Add text inside bars
        for bar, value, word in zip(bars, feature_values, feature_names):
            plt.text(bar.get_width() / 2, bar.get_y() + bar.get_height() / 2,
                     f"{word}", ha='center', va='center', fontsize=12, color='white', weight='bold')

        plt.xlabel("Feature Contribution", fontsize=12)
        plt.ylabel("Words/Phrases", fontsize=12)
        plt.title(f"LIME Explanation for Sample {i + 1}", fontsize=14)
        plt.axvline(x=0, color='black', linewidth=1)
        plt.grid(axis='x', linestyle='--', alpha=0.5)

        # Add legend
        legend_labels = [Patch(facecolor='green', label="Increases Question Probability"),
                         Patch(facecolor='red', label="Decreases Question Probability")]
        plt.legend(handles=legend_labels, loc="lower right", fontsize=10)

        plt.tight_layout()

        # Save and show the figure
        plot_path = os.path.join(output_dir, f'lime_explanation_{i + 1}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()

        # Save explanation as HTML
        exp.save_to_file(os.path.join(output_dir, f'lime_explanation_{i + 1}.html'))

        print(f"LIME explanation saved: {plot_path}")

# Define indices to explain
indices_to_explain = [0, 1]

# Call function
apply_lime_explanation(X, svm_model, tokenized_texts, indices_to_explain)
