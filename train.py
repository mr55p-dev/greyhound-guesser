import sys
import torch
import numpy as np
from torch import backends, mps, nn
from torch.cuda import is_available
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm, trange
import wandb
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


class Network(nn.Module):
    N_DOGS = 6
    DOG_FEATURES = 3
    GLOBAL_FEATURES = 1
    LAYER_1_NODES = 64
    LAYER_2_NODES = 64

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._layers = nn.Sequential(
            nn.Linear(
                self.N_DOGS * self.DOG_FEATURES + self.GLOBAL_FEATURES,
                self.LAYER_1_NODES,
            ),
            nn.ReLU(),
            nn.Linear(self.LAYER_1_NODES, self.LAYER_2_NODES),
            nn.ReLU(),
            nn.Linear(self.LAYER_2_NODES, self.N_DOGS),
            nn.Softmax(dim=0),
        )
        

    def forward(self, x):
        logits = self._layers(x)
        return logits


def train_loop(dataloader, model, loss_func, optimizer, epoch):
    model.train()
    for batch, (X, Y) in enumerate(
        tqdm(dataloader, desc="Train batch", position=1, leave=False)
    ):
        model.zero_grad()

        pred = model.forward(X)
        loss = loss_func(pred, Y)

        loss.backward()
        optimizer.step()

        if batch % 20 == 0:
            wandb.log({"train_loss": loss, "epoch": epoch})


def test_loop(dataloader, model, loss_func, epoch):
    model.eval()
    err = 0
    with torch.no_grad():
        for X, Y in tqdm(dataloader, desc="Test batch", position=1, leave=False):
            pred = model.forward(X)
            err += loss_func(pred, Y).item()

    err /= len(dataloader)
    wandb.log({"test_loss": err, "epoch": epoch})


def main():
    # Hyperparameters
    batch_size = 16
    learning_rate = 1e-3
    epochs = 5

    # Wandb
    wandb.init(
        project="greyhound-guesser",
        config={
            "learning_rate": learning_rate,
            "architecture": "v0",
            "dataset": "crayford-races-v0",
            "epochs": epochs,
            "batch_size": batch_size,
        },
    )
    print("Starting training session")

    # Backedn
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    device = torch.device(device)

    # Data load
    dataset = GreyhoundDataset(device)
    dataloader = DataLoader(dataset, batch_size, shuffle=True)

    # Model load
    model = Network().to(device)
    print("Created model")
    print(model)

    # Training loop
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    for epoch in trange(epochs, desc="Epoch", position=0):
        train_loop(dataloader, model, loss_func, optimizer, epoch)
        test_loop(dataloader, model, loss_func, epoch)

    print("Done!")
    return 0


if __name__ == "__main__":
    wandb.login()
    sys.exit(main())
