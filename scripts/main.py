import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog
from embedding_generation import generate_embeddings
from lime_explanation import apply_lime_explanation
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import os

# Function to allow the user to select a file using a file dialog
def select_input_file():
    root = Tk()
    root.withdraw()  # Hide the root window
    input_file = filedialog.askopenfilename(title="Select the Excel file for prediction",
                                            filetypes=[("Excel files", "*.xlsx")])
    return input_file


# Function to save output in the 'outputs' folder within the project directory
def save_output(df):
    # Ensure the outputs folder exists
    if not os.path.exists('outputs'):
        os.makedirs('outputs')

    # Define the output file path
    output_file = os.path.join('outputs', 'predictions_output.xlsx')

    # Save the DataFrame to an Excel file in the outputs folder
    df.to_excel(output_file, index=False)
    print(f"Predictions saved to {output_file}")


# Function to load the trained SVM model
def load_model(model_path='models/svm_model_bert.pkl'):
    """ Load the trained SVM model from disk. """
    try:
        model = joblib.load(model_path)
        print(f"Model loaded from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


# Function to preprocess data, generate embeddings, and predict labels using the trained model
def preprocess_and_predict(input_file, model):
    """ Preprocess data, generate embeddings, and predict labels using the trained model. """
    # Load the Excel data
    df = pd.read_excel(input_file)

    if 'Text' not in df.columns:
        print("The input file must contain 'Text' column.")
        return

    # Step 1: Preprocess the text and generate BERT embeddings
    df['BERT_Embedding'] = df['Text'].apply(generate_embeddings)

    # Step 2: Prepare features (BERT embeddings) for prediction
    X = np.stack(df['BERT_Embedding'].values)  # Features (BERT embeddings)

    # Step 3: Predict the labels using the trained SVM model
    y_pred = model.predict(X)
    df['Predicted_Label'] = y_pred

    # Step 4: Calculate and display metrics (accuracy, confusion matrix)
    if 'Label' in df.columns:
        y_true = df['Label'].values
        accuracy = accuracy_score(y_true, y_pred)
        print(f"Accuracy: {accuracy:.4f}")

        report = classification_report(y_true, y_pred)
        print("Classification Report:\n", report)

        # Plot and save confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
        disp.plot(cmap="Blues")
        plt.savefig('outputs/confusion_matrix.png')

    # Step 5: Save the output in the 'outputs' folder
    save_output(df)

    # Step 6: Ask the user to select the indices to explain and apply LIME explanation
    indices_to_explain = get_indices_to_explain()
    apply_lime_explanation(X, model, indices_to_explain)


# Function to get indices from the user for LIME explanation
def get_indices_to_explain():
    """ Get indices from the user for which to explain the predictions. """
    indices_input = input("Enter the indices of the samples you want to explain (comma-separated): ")
    indices = [int(i.strip()) for i in indices_input.split(',')]
    return indices


# Main flow
if __name__ == "__main__":
    # Load the pre-trained model
    model = load_model()

    if model:
        # Allow the user to select the input Excel file
        input_file = select_input_file()

        # Run the prediction process
        preprocess_and_predict(input_file, model)
