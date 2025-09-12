import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from mordred import Calculator, descriptors


def calculate_features(smiles_list):
    """Generate RDKit, Morgan FP, and Mordred features from SMILES."""
    # RDKit descriptors
    desc_names = [d[0] for d in Descriptors._descList]
    calc = MoleculeDescriptors.MolecularDescriptorCalculator(desc_names)
    rdkit_data = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        rdkit_data.append(calc.CalcDescriptors(mol) if mol else [0]*len(desc_names))
    rdkit_df = pd.DataFrame(rdkit_data, columns=desc_names)

    # Morgan fingerprints
    n_bits = 2048
    generator = GetMorganGenerator(radius=2, fpSize=n_bits)
    fps_data = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        fps_data.append(list(generator.GetFingerprint(mol)) if mol else [0]*n_bits)
    fps_df = pd.DataFrame(fps_data, columns=[f"ECFP_{i}" for i in range(n_bits)])

    # Mordred descriptors
    calc_mordred = Calculator(descriptors, ignore_3D=True)
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    mordred_df = calc_mordred.pandas(mols, quiet=True)
    mordred_df = mordred_df.apply(pd.to_numeric, errors="coerce").fillna(0)

    # Combine
    full_features_df = pd.concat([rdkit_df, fps_df, mordred_df], axis=1)
    full_features_df = full_features_df.astype(np.float64)
    return full_features_df


def deduplicate_columns(cols):
    """Remove duplicate feature names by appending suffixes."""
    seen = {}
    new_cols = []
    for col in cols:
        if col not in seen:
            seen[col] = 1
            new_cols.append(col)
        else:
            new_name = f"{col}_{seen[col]}"
            while new_name in seen:
                seen[col] += 1
                new_name = f"{col}_{seen[col]}"
            seen[col] += 1
            seen[new_name] = 1
            new_cols.append(new_name)
    return new_cols
