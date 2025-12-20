import numpy as np
import torch
from sklearn.utils import resample

class ImbalanceHandler:
    def __init__(self):
        pass
    def balance_data(self, X_train, y_train):
        # Unimos para remuestrear   
        train_data_np = np.hstack((X_train, y_train.reshape(-1, 1)))
        majority = train_data_np[train_data_np[:, -1] == 0]
        minority = train_data_np[train_data_np[:, -1] == 1]

        # Upsample minoría
        minority_upsampled = resample(minority, replace=True, n_samples=len(majority), random_state=42)
        upsampled_data = np.vstack((majority, minority_upsampled))

        # Nuevos tensores de entrenamiento balanceados
        X_train_res = torch.tensor(upsampled_data[:, :-1], dtype=torch.float32)
        y_train_res = torch.tensor(upsampled_data[:, -1], dtype=torch.float32).unsqueeze(1)
        return X_train_res, y_train_res
