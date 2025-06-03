import os
import argparse
import joblib
import pandas as pd
from src.featurizer import featurize

# Fixed model directory
MODEL_DIR = "/mnt/6dc794ab-96c3-4f0d-97d9-1fa54dbe0b70/The_Prince/ML_Models/OpioidBias/model"
MODEL_PATH = os.path.join(MODEL_DIR, "RandomForest.pkl")
FEATURES_PATH = os.path.join(MODEL_DIR, "selected_features.pkl")

def predict_single_smiles(smiles):
    features = featurize([smiles], selected_features_path=FEATURES_PATH)
    model = joblib.load(MODEL_PATH)
    prob = model.predict_proba(features)[0][1]
    predicted_class = "G-protein biased" if prob >= 0.60 else "β-arrestin biased"
    print(f"Predicted Class: {predicted_class}")
    print(f"Probability of being G-protein biased: {prob:.4f}")

def predict_from_csv(csv_path, output_path):
    df = pd.read_csv(csv_path)
    if 'smiles' not in df.columns:
        raise ValueError("Input CSV must contain a 'smiles' column.")
    
    smiles_list = df['smiles'].tolist()
    features = featurize(smiles_list, selected_features_path=FEATURES_PATH)

    model = joblib.load(MODEL_PATH)
    probabilities = model.predict_proba(features)[:, 1]  # Probabilities for class 1 (G-protein biased)
    classes = ["G-protein biased" if p >= 0.60 else "β-arrestin biased" for p in probabilities]

    df['Predicted_Class'] = classes
    df['G_Protein_Probability'] = probabilities

    df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Predict B-arrestin vs G-protein biased ligands.")
    parser.add_argument("--smiles", type=str, help="SMILES string of a single molecule")
    parser.add_argument("--csv", type=str, help="Path to CSV file containing a 'smiles' column")
    parser.add_argument("--output", type=str, default="predictions.csv", help="Output CSV file name (for --csv mode)")

    args = parser.parse_args()

    if args.smiles:
        predict_single_smiles(args.smiles)
    elif args.csv:
        predict_from_csv(args.csv, args.output)
    else:
        print("Please provide either a --smiles input or a --csv file.")
        exit(1)

if __name__ == "__main__":
    main()
