import joblib
import pandas as pd
import numpy as np
from embedding_generation import generate_embeddings
import os


def predict_labels(input_file):
    """ Function to predict labels for the input file and save results to Excel """
    # Load the Excel data (already preprocessed and with embeddings)
    df = pd.read_excel(input_file)

    if 'Text' not in df.columns:
        print("The input file must contain 'Text' column.")
        return

    # Step 1: Generate embeddings for the input data (this will be done in main.py, no need here)
    # df['BERT_Embedding'] = df['Text'].apply(generate_embeddings)

    # Step 2: Load the pre-trained SVM model
    model_path = 'models/svm_model_bert.pkl'
    svm_model = joblib.load(model_path)

    # Step 3: Prepare the embeddings as features (embeddings should already exist)
    X = np.stack(df['BERT_Embedding'].values)

    # Step 4: Predict labels using the trained SVM model
    df['Predicted_Label'] = svm_model.predict(X)

    # Step 5: Save the output in the 'outputs' folder
    save_output(df)


def save_output(df):
    """ Function to save the output to an Excel file in the 'outputs' folder """
    # Ensure the outputs folder exists
    if not os.path.exists('outputs'):
        os.makedirs('outputs')

    # Define the output file path
    output_file = os.path.join('outputs', 'predictions_output.xlsx')

    # Save the DataFrame to an Excel file in the outputs folder
    df.to_excel(output_file, index=False)
    print(f"Predictions saved to {output_file}")
