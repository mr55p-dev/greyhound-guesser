import sys
import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm, trange
import wandb
from src.model import Network
from src.dataset import GreyhoundDataset

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
