import yaml
import os

class Config:
    def __init__(self, path='configurations.yaml'):
        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        self.RANDOM_STATE = data['random_state']
        self.DATASET = data['dataset']
        self.PREPROCESSING = data['preprocessing']
        self.FEATURES = data['features']
        self.SAMPLING = data['sampling']
        self.MODELS = data['models']
        self.EVALUATION = data['evaluation']

        self._create_dirs()

    # create dir for processed data and trained models
    def _create_dirs(self):
        dirs = [
            self.DATASET['prepared']['dir'],
            self.DATASET['sampled']['dir'],
            self.MODELS['dir'],
            self.MODELS['logistic_regression']['dir'],
            self.MODELS['random_forest']['dir'],
            self.MODELS['voting_classifier']['dir'],
            self.MODELS['neural_network']['dir']
        ]
        for d in dirs:
            os.makedirs(d, exist_ok=True)

    def __repr__(self):
        return (f"Config(random_state={self.RANDOM_STATE}, "
                f"dataset_keys={list(self.DATASET.keys())})")

config = Config()   # create config