import joblib
import numpy as np
from src.featurizer import calc_features, get_fingerprint

class BiasPredictor:
    def __init__(self, model_path):
        self.model = joblib.load(model_path)

    def preprocess(self, smiles):
        smiles_list = [smiles]
        features_df = featurize(smiles_list)

        if features_df.isnull().values.any():
            raise ValueError("Invalid SMILES or feature computation failed.")
        return features_df.values


    def predict(self, smiles):
        X = self.preprocess(smiles)
        return self.model.predict(X)[0]
