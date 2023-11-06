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
        data["Finish_Recent"] = (6 - data["Finish_Recent"]) / 5

        races = data["Race_ID"].unique()
        features = np.empty((19, 1, len(races)), dtype=np.float32)
        places = np.zeros((6, 1, len(races)), dtype=np.float32)
        labels = np.zeros(places.shape, dtype=np.float32)

        features[0, 0, :] = 380 / 1000

        for idx, race in enumerate(races):
            dogs = data[data["Race_ID"] == race].sort_values("Trap")
            dogs = dogs[
                ["Odds", "Distance_Recent", "Finish_Recent", "Finished"]
            ].reset_index(drop=True)
            dog_features = dogs.drop("Finished", axis=1).values.reshape(18, 1)
            dog_labels = dogs["Finished"].values.reshape(6, 1)
            features[1:, 0, idx] = dog_features[:, 0]
            places[:, 0, idx] = dog_labels[:, 0]

        np.put_along_axis(
            labels,
            np.argmin(places, 0, keepdims=True),
            1,
            axis=0,
        )

        self._data = data
        self._features = torch.tensor(features, device=device)
        self._labels = torch.tensor(labels, device=device)

    def __len__(self) -> int:
        return self._features.shape[2]

    def __getitem__(self, index):
        feature = self._features[:, 0, index]
        label = self._labels[:, 0, index]
        return feature, label
