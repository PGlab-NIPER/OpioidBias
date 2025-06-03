from src.predictor import BiasPredictor

def test_prediction():
    predictor = BiasPredictor("model/best_model.pkl")
    smiles = "CCO"  # Simple ethanol molecule
    result = predictor.predict(smiles)
    assert result in [0, 1]
