import argparse
import warnings
import joblib
import pickle
import pandas as pd
from src.featurizer import calculate_features, deduplicate_columns

# Suppress NumPy warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


def load_model_and_scaler():
    """Load model, scaler, and feature sets."""
    model = joblib.load("model/RandomForest.joblib")
    scaler = joblib.load("model/minmax_scaler.pkl")
    rfe_features = pd.read_csv("features/selected_features.csv")  # CSV version
    features_for_scaling = pickle.load(open("features/features_for_scaling.pkl", "rb"))
    return model, scaler, rfe_features, features_for_scaling


def prepare_and_predict(df):
    """Run preprocessing and prediction pipeline."""
    model, scaler, rfe_features, features_for_scaling = load_model_and_scaler()
    smiles_list = df["SMILES"].tolist()
    ligand_names = df["Ligand"].tolist()

    # Extract features
    full_features_df = calculate_features(smiles_list)
    full_features_df.columns = deduplicate_columns(full_features_df.columns)

    # Reorder columns for scaler
    full_features_df = full_features_df.reindex(columns=features_for_scaling, fill_value=0)

    # Scale features
    scaled_array = scaler.transform(full_features_df)
    scaled_df = pd.DataFrame(scaled_array, columns=features_for_scaling)

    # Select RFE features
    rfe_features_list = rfe_features["RFE_Features"].astype(str).tolist()
    X_rfe = scaled_df[rfe_features_list]

    # Predict
    preds = model.predict(X_rfe)
    probs = model.predict_proba(X_rfe)[:, 1] if hasattr(model, "predict_proba") else preds
    class_labels = ["Beta-arrestin" if p == 0 else "G-Protein" for p in preds]

    results_df = pd.DataFrame({
        "Ligand": ligand_names,
        "SMILES": smiles_list,
        "Predicted_Class": class_labels,
        "Probability_G-Protein": probs
    })
    return results_df


def main():
    parser = argparse.ArgumentParser(description="Predict B-arrestin vs G-protein biased ligands.")
    parser.add_argument("--smiles", type=str, help="SMILES string of a single molecule")
    parser.add_argument("--csv", type=str, help="Path to CSV file containing a 'SMILES' column")
    parser.add_argument("--output", type=str, default="predictions.csv", help="Output CSV file name (for --csv mode)")

    args = parser.parse_args()

    if args.smiles:
        df = pd.DataFrame({"SMILES": [args.smiles], "Ligand": ["Ligand_1"]})
        results = prepare_and_predict(df)
        print(results.to_string(index=False))

    elif args.csv:
        df = pd.read_csv(args.csv)
        if "SMILES" not in df.columns:
            raise ValueError("Input CSV must contain a 'SMILES' column.")
        if "Ligand" not in df.columns:
            df["Ligand"] = [f"Ligand_{i}" for i in range(len(df))]
        results = prepare_and_predict(df)
        results.to_csv(args.output, index=False)
        print(f"Predictions saved to {args.output}")

    else:
        parser.error("Please provide either --smiles or --csv")


if __name__ == "__main__":
    main()
