import yaml
import os

class Config:
    def __init__(self, path='configurations.yaml'):
        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        self.RANDOM_STATE = data['random_state']
        self.SEED = data['seed']
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
            self.DATASET['prepared']['train']['dir'],
            self.DATASET['prepared']['val']['dir'],
            self.DATASET['prepared']['train_val']['dir'],
            self.DATASET['prepared']['test']['dir'],
            self.DATASET['sampled']['dir'],
            self.DATASET['sampled']['rus']['dir'],
            self.DATASET['sampled']['enn']['dir'],
            self.DATASET['sampled']['smote']['dir'],
            self.DATASET['sampled']['smoteenn']['dir'],
            self.DATASET['sampled']['smotetomek']['dir'],
            self.MODELS['dir'],
            self.MODELS['logistic_regression']['dir'],
            self.MODELS['logistic_regression']['sample']['none']['dir'],
            self.MODELS['logistic_regression']['sample']['rus']['dir'],
            self.MODELS['logistic_regression']['sample']['enn']['dir'],
            self.MODELS['logistic_regression']['sample']['smote']['dir'],
            self.MODELS['logistic_regression']['sample']['smoteenn']['dir'],
            self.MODELS['logistic_regression']['sample']['smotetomek']['dir'],
            self.MODELS['random_forest']['dir'],
            self.MODELS['random_forest']['sample']['none']['dir'],
            self.MODELS['random_forest']['sample']['rus']['dir'],
            self.MODELS['random_forest']['sample']['enn']['dir'],
            self.MODELS['random_forest']['sample']['smote']['dir'],
            self.MODELS['random_forest']['sample']['smoteenn']['dir'],
            self.MODELS['random_forest']['sample']['smotetomek']['dir'],
            self.MODELS['neural_network']['dir'],
            self.MODELS['neural_network']['sample']['none']['dir'],
            self.MODELS['neural_network']['sample']['rus']['dir'],
            self.MODELS['neural_network']['sample']['enn']['dir'],
            self.MODELS['neural_network']['sample']['smote']['dir'],
            self.MODELS['neural_network']['sample']['smoteenn']['dir'],
            self.MODELS['neural_network']['sample']['smotetomek']['dir'],
            self.MODELS['neural_network_fl']['dir'],
            self.MODELS['neural_network_fl']['sample']['none']['dir'],
            self.MODELS['neural_network_fl']['sample']['rus']['dir'],
            self.MODELS['neural_network_fl']['sample']['enn']['dir'],
            self.MODELS['neural_network_fl']['sample']['smote']['dir'],
            self.MODELS['neural_network_fl']['sample']['smoteenn']['dir'],
            self.MODELS['neural_network_fl']['sample']['smotetomek']['dir'],
            self.MODELS['voting_classifier']['dir'],
            self.MODELS['voting_classifier']['sample']['none']['dir'],
            self.MODELS['voting_classifier']['sample']['rus']['dir'],
            self.MODELS['voting_classifier']['sample']['enn']['dir'],
            self.MODELS['voting_classifier']['sample']['smote']['dir'],
            self.MODELS['voting_classifier']['sample']['smoteenn']['dir'],
            self.MODELS['voting_classifier']['sample']['smotetomek']['dir']
        ]
        for d in dirs:
            os.makedirs(d, exist_ok=True)

    def __repr__(self):
        return (f"Config(random_state={self.RANDOM_STATE}, "
                f"dataset_keys={list(self.DATASET.keys())})")
