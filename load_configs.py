import yaml
import os

class Config:
    def __init__(self, path='configs.yml'):
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
            self.DATASET['sampled']['dir'],
            self.DATASET['sampled']['rus']['dir'],
            self.DATASET['sampled']['enn']['dir'],
            self.DATASET['sampled']['smote']['dir'],
            self.DATASET['sampled']['smoteenn']['dir'],
            self.DATASET['sampled']['smotetomek']['dir'],
            self.DATASET['sampled']['rus']['train']['dir'],
            self.DATASET['sampled']['enn']['train']['dir'],
            self.DATASET['sampled']['smote']['train']['dir'],
            self.DATASET['sampled']['smoteenn']['train']['dir'],
            self.DATASET['sampled']['smotetomek']['train']['dir'],
            self.DATASET['sampled']['rus']['train_val']['dir'],
            self.DATASET['sampled']['enn']['train_val']['dir'],
            self.DATASET['sampled']['smote']['train_val']['dir'],
            self.DATASET['sampled']['smoteenn']['train_val']['dir'],
            self.DATASET['sampled']['smotetomek']['train_val']['dir'],
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
            self.MODELS['knn_classifier']['dir'],
            self.MODELS['knn_classifier']['sample']['none']['dir'],
            self.MODELS['knn_classifier']['sample']['rus']['dir'],
            self.MODELS['knn_classifier']['sample']['enn']['dir'],
            self.MODELS['knn_classifier']['sample']['smote']['dir'],
            self.MODELS['knn_classifier']['sample']['smoteenn']['dir'],
            self.MODELS['knn_classifier']['sample']['smotetomek']['dir'],
            self.MODELS['voting_classifier_1']['dir'],
            self.MODELS['voting_classifier_1']['sample']['none']['dir'],
            self.MODELS['voting_classifier_1']['sample']['rus']['dir'],
            self.MODELS['voting_classifier_1']['sample']['enn']['dir'],
            self.MODELS['voting_classifier_1']['sample']['smote']['dir'],
            self.MODELS['voting_classifier_1']['sample']['smoteenn']['dir'],
            self.MODELS['voting_classifier_1']['sample']['smotetomek']['dir'],
            self.MODELS['voting_classifier_2']['dir'],
            self.MODELS['voting_classifier_2']['sample']['none']['dir'],
            self.MODELS['voting_classifier_2']['sample']['rus']['dir'],
            self.MODELS['voting_classifier_2']['sample']['enn']['dir'],
            self.MODELS['voting_classifier_2']['sample']['smote']['dir'],
            self.MODELS['voting_classifier_2']['sample']['smoteenn']['dir'],
            self.MODELS['voting_classifier_2']['sample']['smotetomek']['dir'],
            self.MODELS['voting_classifier_3']['dir'],
            self.MODELS['voting_classifier_3']['sample']['none']['dir'],
            self.MODELS['voting_classifier_3']['sample']['rus']['dir'],
            self.MODELS['voting_classifier_3']['sample']['enn']['dir'],
            self.MODELS['voting_classifier_3']['sample']['smote']['dir'],
            self.MODELS['voting_classifier_3']['sample']['smoteenn']['dir'],
            self.MODELS['voting_classifier_3']['sample']['smotetomek']['dir'],
            self.MODELS['best_model']['dir'],
            self.EVALUATION['dir']
        ]
        for d in dirs:
            os.makedirs(d, exist_ok=True)

    def __repr__(self):
        return (f"Config(random_state={self.RANDOM_STATE}, "
                f"dataset_keys={list(self.DATASET.keys())})")
