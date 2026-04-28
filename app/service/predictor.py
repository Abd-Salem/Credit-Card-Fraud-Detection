import joblib
import pandas as pd
from load_configs import Config

# features name
FEATURE_COLUMNS = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
        'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
        'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']

# pipeline service
class ModelService:
    def __init__(self, path:str=Config().MODELS['best_model']['model'], threshold=0.5, name='voting_classifier'):
        self.model = joblib.load(path)
        self.threshold = threshold
        self.name = name

    def predict(self, X):
        prob = self.model.predict_proba(X)[:,1]
        pred = (prob >= self.threshold).astype(int)
        return prob, pred

# prepare input for pipeline
def prepare_input(features: list[float]):
    return pd.DataFrame([features], columns=FEATURE_COLUMNS)