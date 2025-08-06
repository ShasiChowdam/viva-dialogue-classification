# svm_model_training.py
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import joblib
import numpy as np
import matplotlib.pyplot as plt
from embedding_generation import generate_embeddings


def train_svm_model(input_file):
    """ Train an SVM model on the input data and save the model. """

    # Load data
    df = pd.read_excel(input_file)

    # Check if required columns are present
    if 'Text' not in df.columns or 'Label' not in df.columns:
        print("The input file must contain 'Text' and 'Label' columns.")
        return

    # Step 1: Preprocess the text and generate BERT embeddings
    df['embedding'] = df['Text'].apply(generate_embeddings)

    # Step 2: Prepare features and labels
    X = np.stack(df['embedding'].values)  # Features (BERT embeddings)
    y = df['Label'].values  # Labels (Question, Answer, Statement)

    # Step 3: Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Step 4: Train the SVM model with probability=True
    svm_model = SVC(kernel='linear', C=1, probability=True)  # Enable probability
    svm_model.fit(X_train, y_train)

    # Step 5: Save the model
    model_path = 'models/svm_model_bert.pkl'
    joblib.dump(svm_model, model_path)
    print(f"Model trained and saved to {model_path}")

    # Step 6: Evaluate the model
    y_pred = svm_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")

    report = classification_report(y_test, y_pred)
    print("Classification Report:\n", report)

    # Save classification report
    with open('classification_report.txt', 'w') as f:
        f.write(report)

    # Plot and save confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=svm_model.classes_)
    disp.plot(cmap="Blues")
    plt.savefig('confusion_matrix.png')


if __name__ == "__main__":
    input_file = r"C:\Users\anany\PycharmProjects\viva_seperation\data\Final Viva Text.xlsx" # Provide the path to your training data
    train_svm_model(input_file)
