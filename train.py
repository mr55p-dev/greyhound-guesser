from datetime import datetime
import sys
import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import random_split
from tqdm import trange
import wandb
from src.model import Network
from src.dataset import GreyhoundDataset, BATCH_SIZE
from src.utils import get_device
from src.metrics import Accuracy, Metric


def train_loop(
    dataloader, model, loss_func, optimizer, epoch, metrics: list[Metric] = []
):
    model.train()
    for batch, (X, Y) in enumerate(dataloader):
        model.zero_grad()

        pred = model.forward(X)
        loss = loss_func(pred, Y)
        for metric in metrics:
            metric.log(pred, Y)

        loss.backward()
        optimizer.step()

        metric_log = {metric.get_name(): metric.reset() for metric in metrics}
        if batch % 20 == 0:
            wandb.log(
                {
                    "train_loss": loss,
                    "epoch": epoch,
                    **metric_log,
                }
            )


def test_loop(dataloader, model, loss_func, epoch, metrics: list[Metric] = []):
    model.eval()
    err = 0

    with torch.no_grad():
        for X, Y in dataloader:
            pred = model.forward(X)
            err += loss_func(pred, Y).item()
            for metric in metrics:
                metric.log(pred, Y)

    err /= len(dataloader)
    metric_log = {metric.get_name(): metric.reset() for metric in metrics}

    wandb.log(
        {
            "test_loss": err,
            "epoch": epoch,
            **metric_log,
        }
    )


def main():
    # Hyperparameters
    learning_rate = 1e-5
    epochs = 5000

    torch.manual_seed(1)

    # Wandb
    wandb.init(
        project="greyhound-guesser",
        config={
            "learning_rate": learning_rate,
            "architecture": "v0.1",
            "dataset": "crayford-races-v0",
            "epochs": epochs,
            "batch_size": BATCH_SIZE,
            "optimizer": "Adam",
        },
    )
    print("Starting training session")

    # Backend
    device = get_device()

    # Data load
    dataset = GreyhoundDataset(device)
    train_set, test_set = random_split(dataset, [0.9, 0.1])
    train = DataLoader(train_set, BATCH_SIZE, shuffle=True)
    test = DataLoader(test_set, BATCH_SIZE, shuffle=True)

    # Model load
    model = Network().to(device)
    print("Created model")
    print(model)

    # Metrics
    train_metrics: list[Metric] = [
        Accuracy(len(train), name="train_accuracy"),
    ]

    test_metrics: list[Metric] = [
        Accuracy(len(test), name="test_accuracy"),
    ]

    # Training loop
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in trange(epochs, desc="Epoch", position=0):
        train_loop(train, model, loss_func, optimizer, epoch, train_metrics)
        test_loop(test, model, loss_func, epoch, test_metrics)

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
