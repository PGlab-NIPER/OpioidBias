# OpioidBias Predictor

This tool predicts whether opioids targeting agonists are **G-protein biased** or **β-arrestin biased** based on their SMILES representations using a trained Random Forest model. 
It supports both **single-molecule** prediction and **batch prediction via CSV**.

---

## 🧪 Model Details

- **Model type**: Random Forest
- **Input**: RDKit-based descriptors and molecular fingerprints
- **Threshold**: Molecules with probability ≥ 0.60 are classified as **G-protein biased**; otherwise, as **β-arrestin biased**

---

## 🔧 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Rajkumar-Raja/OpioidBias.git
   cd OpioidBias

2. Create the conda environment:
   conda env create -f environment.yml
   conda activate opioidbias

3. Usage:
   i) Predict a Single SMILES

   python main.py --smiles "CCN(CC)CCCC(C)Nc1ccc2c(c1)C(=O)N(C3CCC(CC3)NC(=O)OC(C)(C)C)C2=O"

   Example Output:
   Predicted Class: G-protein biased
   Probability of being G-protein biased: 0.7634

   ii)  Predict from a CSV File

   python predict.py --csv input.csv --output predictions.csv

4. Project Structure:
   OpioidBias/
   │
   ├── predict.py                 # Main script
   ├── src/
   │   └── featurizer.py          # Feature extraction code
   ├── model/
   │   ├── RandomForest.pkl       # Trained classifier
   │   └── selected_features.pkl  # Selected features used by the model
   ├── environment.yml            # Conda environment configuration
   └── README.md

5.Citation:
  If you use this tool in your work, please cite:

  "OpioidBias: A Machine Learning Framework for Predicting Biased Agonism of Opioid Ligands"

Contact
For questions or feedback, please contact:
Rajkumar Raja
rajkumarrpi22@gmail.com
