import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import joblib
import numpy as np
import matplotlib.pyplot as plt
from embedding_generation import generate_embeddings

def train_svm_model(input_file):
    # Load data
    df = pd.read_excel(input_file)

    if 'Text' not in df.columns or 'Label' not in df.columns:
        print("The input file must contain 'Text' and 'Label' columns.")
        return

    # Step 1: Preprocess the text and generate BERT embeddings
    df = generate_embeddings(df)  # Generate embeddings for the cleaned text

    # Step 2: Prepare features and labels
    X = np.stack(df['embedding'].values)  # Features (BERT embeddings)
    y = df['Label'].values  # Labels (Question, Answer, Statement)

    # Step 3: Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Step 4: Train the SVM model
    svm_model = SVC(kernel='linear', C=1)
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
    with open('outputs/classification_report.txt', 'w') as f:
        f.write(report)

    # Plot and save confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=svm_model.classes_)
    disp.plot(cmap="Blues")
    plt.savefig('outputs/confusion_matrix.png')
