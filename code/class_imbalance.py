import numpy as np
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

class ImbalanceHandler:
    def __init__(self):
        pass

    def upsample_data(self, X_train, y_train):
        """
        Aplica Random Over Sampling.
        Devuelve arrays de Numpy compatibles con sklearn.
        """
        ros = RandomOverSampler(random_state=42)
        X_res, y_res = ros.fit_resample(X_train, y_train)
        return X_res, y_res
    
    def downsample_data(self, X_train, y_train):
        """
        Aplica Random Under Sampling.
        Devuelve arrays de Numpy compatibles con sklearn.
        """
        rus = RandomUnderSampler(random_state=42)
        X_res, y_res = rus.fit_resample(X_train, y_train)
        return X_res, y_res

    def smote_data(self, X_train, y_train, k_neighbors=5):
        """
        Aplica SMOTE.
        Devuelve arrays de Numpy compatibles con sklearn.
        """
        # SMOTE maneja automáticamente si k_neighbors > n_samples_minority
        smote = SMOTE(k_neighbors=k_neighbors, random_state=42)
        X_res, y_res = smote.fit_resample(X_train, y_train)
        return X_res, y_res