import sys
import torch
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
from pathlib import Path


class GreyhoundDataset(Dataset):
    LABEL_COL = "Winner"

    def __init__(self, device) -> None:
        data = Path("./data/crayford-races.csv")
        if not data.exists():
            print("No data found")
            sys.exit(1)

        data = pd.read_csv(data)
        data["Odds"] = 1 / data["Odds"]
        data["Distance_Recent"] = data["Distance_Recent"] / 1000
        data["Finish_Recent"] = (6 - data["Finish_Recent"].clip(lower=1, upper=6)) / 5
        all_cols = ["Odds", "Distance_Recent", "Finish_Recent", "Finished"]
        feature_cols = ["Odds", "Distance_Recent", "Finish_Recent"]

        races = data["Race_ID"].unique()
        n = len(races)

        features = np.empty((n, 19), dtype=np.float32)
        places = np.zeros((n, 6), dtype=np.float32)
        labels = np.zeros(places.shape, dtype=np.float32)

        # Preset race length
        features[:, 0] = 380 / 1000

        for idx, race in enumerate(races):
            # Get all dogs running for this race sorted by trap from 1 -> 6
            dogs = data[data["Race_ID"] == race].sort_values("Trap")
            dogs = dogs[all_cols].reset_index(drop=True)

            # Generate the feature and label matrices
            features[idx, 1:] = dogs[feature_cols].values.reshape(18)
            places[idx, :] = dogs["Finished"].values

        np.put_along_axis(
            labels,
            np.argmin(places, 1, keepdims=True),
            1,
            axis=1,
        )

        self._data = data
        self._features = torch.tensor(features, device=device)
        self._labels = torch.tensor(labels, device=device)

    def __len__(self) -> int:
        return self._features.shape[0]

    def __getitem__(self, index):
        feature = self._features[index, :]
        label = self._labels[index, :]
        return feature, label
