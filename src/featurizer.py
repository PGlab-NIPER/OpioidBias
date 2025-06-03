import pandas as pd
import pickle
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from sklearn.preprocessing import MinMaxScaler

# ---- Step 1: RDKit Descriptors ----
def calculate_descriptors(smiles_list):
    descriptor_names = [desc[0] for desc in Descriptors._descList]
    calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)
    descriptor_data = []

    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            descriptors = calculator.CalcDescriptors(mol)
        else:
            print(f"Warning: Invalid SMILES - {smiles}")
            descriptors = [0] * len(descriptor_names)
        descriptor_data.append(descriptors)

    return pd.DataFrame(descriptor_data, columns=descriptor_names)

# ---- Step 2: Morgan Fingerprints ----
def calculate_morgan_fingerprints(smiles_list, radius=2, n_bits=2048):
    generator = GetMorganGenerator(radius=radius, fpSize=n_bits)
    fingerprints = []

    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            fp = generator.GetFingerprint(mol)
            fingerprints.append(list(fp))
        else:
            print(f"Warning: Failed to parse SMILES - {smiles}")
            fingerprints.append([0] * n_bits)  # Explicitly 0s for invalid
    col_names = [f"FP_{i}" for i in range(n_bits)]
    return pd.DataFrame(fingerprints, columns=col_names)

# ---- Step 3: Full Featurizer ----
def featurize(smiles_list, selected_features_path='selected_features.pkl'):
    # Calculate descriptors and fingerprints
    desc_df = calculate_descriptors(smiles_list)
    morgan_df = calculate_morgan_fingerprints(smiles_list)
    combined_df = pd.concat([desc_df, morgan_df], axis=1)

    # ---- Step 4: Apply MinMax Scaling to all features ----
    #scaler = MinMaxScaler()
    #scaled_array = scaler.fit_transform(combined_df.fillna(0))  # Fill NaNs for scaling
    #scaled_df = pd.DataFrame(scaled_array, columns=combined_df.columns)

    # ---- Step 5: Load selected features ----
    with open(selected_features_path, 'rb') as f:
        selected_features = pickle.load(f)

    # Filter only selected features
    filtered_df = combined_df[selected_features]

    #print("Final features preview:")
    #print(filtered_df.head(1))  # Show one row
    #print("Sum of features:", filtered_df.sum(axis=1).values)  # Should NOT be 0

    
    return filtered_df


