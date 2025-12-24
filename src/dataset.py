import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

class MacroDataset(Dataset):
    def __init__(self, df, feature_cols, target_col):
        self.features = torch.tensor(df[feature_cols].values, dtype=torch.float32)
        self.targets = torch.tensor(df[target_col].values, dtype=torch.float32).unsqueeze(-1)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


def load_macro_data(csv_path, feature_cols, target_col, train_frac=0.7, val_frac=0.15):
    df = pd.read_csv(csv_path)
    df = df.sort_values('date')  # very important: time order

    # basic cleaning example (customize as needed)
    df = df.dropna()

    n = len(df)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    n_test = n - n_train - n_val

    df_train = df.iloc[:n_train]
    df_val = df.iloc[n_train:n_train+n_val]
    df_test = df.iloc[n_train+n_val:]

    train_ds = MacroDataset(df_train, feature_cols, target_col)
    val_ds = MacroDataset(df_val, feature_cols, target_col)
    test_ds = MacroDataset(df_test, feature_cols, target_col)

    return train_ds, val_ds, test_ds, df