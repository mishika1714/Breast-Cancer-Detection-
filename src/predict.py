import os
import joblib
import pandas as pd

class CancerInferenceEngine:
    def __init__(self, model_path: str = "models/champion_breast_cancer_model.joblib"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}. Please run train_model.py first.")
        self.pipeline = joblib.load(model_path)
        # Store expected feature array columns from the scaler context
        self.feature_names = self.pipeline.steps[0][1].feature_names_in_

    def predict_observation(self, feature_dict: dict) -> dict:
        """
        Ingests a dictionary of metrics, matches order, and returns diagnostic results.
        """
        input_df = pd.DataFrame([feature_dict])
        # Ensure all columns exist and match exactly in position
        input_df = input_df[self.feature_names]
        
        prediction = self.pipeline.predict(input_df)[0]
        probability = self.pipeline.predict_proba(input_df)[0][1]
        
        return {
            "Diagnosis_Code": int(prediction),
            "Diagnosis_Label": "Malignant" if prediction == 1 else "Benign",
            "Malignant_Probability": float(probability)
        }