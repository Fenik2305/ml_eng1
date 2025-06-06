import logging

from pipeline.download import download_and_extract
from pipeline.ingestion import process_data
from pipeline.loader import create_data_loader
from pipeline.models import ResNetClassifier
from pipeline.training import train_model
from pipeline.testing import test_model

import torch
from torch import nn
from torch import optim


def main():
    logging.basicConfig(level=logging.DEBUG)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = {
        'test_size': 0.2,
        'val_size': 0.2,
        'random_state': 42,
        "lr": 0.001
    }
    
    cifar10_url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    cifar10_dir = download_and_extract(cifar10_url, "./data/")
    train_df, val_df, test_df = process_data(cifar10_dir, config)
    train_loader = create_data_loader(train_df, config)
    val_loader = create_data_loader(val_df, config)
    test_loader = create_data_loader(test_df, config)

    model = ResNetClassifier(n_classes=10).to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config["lr"])

    best_model_path = train_model(
        model, train_loader, val_loader, loss_function, optimizer, num_epochs=15, device=device)

    test_model(model, test_loader, loss_function, device)

if __name__ == "__main__":
    main()
