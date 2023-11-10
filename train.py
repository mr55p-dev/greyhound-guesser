from datetime import datetime
import sys
import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import random_split
from tqdm import tqdm, trange
import wandb
from src.model import Network
from src.dataset import GreyhoundDataset
from src.utils import get_device


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
    accuracy = 0
    with torch.no_grad():
        for X, Y in tqdm(dataloader, desc="Test batch", position=1, leave=False):
            pred = model.forward(X)
            err += loss_func(pred, Y).item()
            is_correct = torch.eq(
                torch.argmax(pred, dim=1),
                torch.argmin(Y, dim=1),
            )
            accuracy += torch.nonzero(is_correct).item()

    err /= len(dataloader)
    accuracy /= len(dataloader)

    wandb.log({
        "test_loss": err,
        "accuracy": accuracy,
        "epoch": epoch,
    })


def main():
    # Hyperparameters
    batch_size = 32
    learning_rate = 1e-4
    epochs = 50

    torch.manual_seed(1)

    # Wandb
    wandb.init(
        project="greyhound-guesser",
        config={
            "learning_rate": learning_rate,
            "architecture": "v0.1",
            "dataset": "crayford-races-v0",
            "epochs": epochs,
            "batch_size": batch_size,
            "optimizer": "Adam",
        },
    )
    print("Starting training session")

    # Backend
    device = get_device()

    # Data load
    dataset = GreyhoundDataset(device)
    train_set, test_set = random_split(dataset, [0.9, 0.1])
    train = DataLoader(train_set, batch_size, shuffle=True)
    test = DataLoader(test_set, batch_size, shuffle=True)

    # Model load
    model = Network().to(device)
    print("Created model")
    print(model)

    # Training loop
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in trange(epochs, desc="Epoch", position=0):
        train_loop(train, model, loss_func, optimizer, epoch)
        test_loop(test, model, loss_func, epoch)

    # Save
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M")
    model_name = f"gg-{ts}.pt"
    model_path = f"models/{model_name}"
    torch.save(model.state_dict(), model_path)

    # Save artifact
    artifact = wandb.Artifact(name=model_name, type="model-state-dict")
    artifact.add_file(model_path)
    wandb.log_artifact(artifact)

    return 0


if __name__ == "__main__":
    wandb.login()
    sys.exit(main())
