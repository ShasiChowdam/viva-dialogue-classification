from limey.lime_tabular import LimeTabularExplainer
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from embedding_generation import generate_embeddings  # Assuming your embedding generation is here
from matplotlib.patches import Patch

# Ensure interactive mode is off for better control over plot display
plt.ioff()

# Function to apply LIME explanation with improved visualization
def apply_lime_explanation(X, model, tokenized_texts, indices_to_explain):
    """ Apply LIME explanation for selected indices with enhanced visualization. """

    # Ensure output directory exists
    output_dir = "outputs"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Initialize LIME explainer
    explainer = LimeTabularExplainer(
        training_data=X,  # Use embeddings as training data
        mode='classification',
        class_names=model.classes_,
        discretize_continuous=True
    )

    for i in indices_to_explain:
        if i >= len(X):
            print(f"Index {i} out of bounds. Skipping.")
            continue

        # Get LIME explanation for the instance
        exp = explainer.explain_instance(X[i], model.predict_proba, num_features=10)

        # Extract feature importance
        exp_map = exp.as_map()[1]  # Get feature contributions
        feature_indices = [idx for idx, _ in exp_map]  # Indices of features
        feature_values = [v for _, v in exp_map]  # Importance values

        # Retrieve actual words using tokenized_texts
        feature_names = [tokenized_texts[i][idx] if idx < len(tokenized_texts[i]) else f"Feature {idx}"
                         for idx in feature_indices]

        # Sort features by absolute importance
        sorted_indices = np.argsort(np.abs(feature_values))[::-1]
        feature_names = [feature_names[idx] for idx in sorted_indices]
        feature_values = [feature_values[idx] for idx in sorted_indices]

        # Define colors (green = question, red = not question)
        colors = ['green' if v > 0 else 'red' for v in feature_values]

        # Generate the improved visualization
        plt.figure(figsize=(10, 6))
        bars = plt.barh(feature_names, feature_values, color=colors, edgecolor="black")

        # Add text inside bars (show words clearly)
        for bar, value, word in zip(bars, feature_values, feature_names):
            plt.text(bar.get_width() / 2, bar.get_y() + bar.get_height() / 2,
                     f"{word}", ha='center', va='center', fontsize=12, color='white', weight='bold')

        plt.xlabel("Feature Contribution to Question Classification", fontsize=12)
        plt.ylabel("Words/Phrases", fontsize=12)
        plt.title(f"LIME Explanation for Sample {i + 1}", fontsize=14)
        plt.axvline(x=0, color='black', linewidth=1)  # Middle reference line
        plt.grid(axis='x', linestyle='--', alpha=0.5)

        # Add legend
        legend_labels = [Patch(facecolor='green', label="Increases Question Probability"),
                         Patch(facecolor='red', label="Decreases Question Probability")]
        plt.legend(handles=legend_labels, loc="lower right", fontsize=10)

        plt.tight_layout()  # Optimize layout

        # Save the figure
        plot_path = os.path.join(output_dir, f'lime_explanation_{i + 1}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')

        # Ensure plot is displayed
        plt.show(block=True)
        plt.pause(0.1)  # Pause to allow rendering in some environments

        # Save explanation as HTML for further review
        exp.save_to_file(os.path.join(output_dir, f'lime_explanation_{i + 1}.html'))

        print(f"LIME explanation saved: {plot_path}")

# Example Usage
if __name__ == "__main__":
    # Load trained model
    model = joblib.load("model.pkl")  # Ensure you have a trained model

    # Load input data (assumed embedding format)
    X = np.load("embeddings.npy")  # Load embeddings from saved file
    tokenized_texts = [["What", "is", "your", "name", "?"],
                       ["This", "is", "a", "test", "sentence", "."]]  # Tokenized words

    # Define indices to explain
    indices_to_explain = [0, 1]  # Select samples for LIME

    # Call function with correct parameters
    apply_lime_explanation(X, model, tokenized_texts, indices_to_explain)
