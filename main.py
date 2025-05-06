import lightning
from lightning.pytorch.loggers import TensorBoardLogger


import torch
import argparse
from model import FeedForwardModel, TwoLayerModel
import yaml
from config import Config
from data import get_mnist_dataloaders

def train_single_model(cfg, train_loader, test_loader):

    # Initialize the model
    if cfg.model_type == 'feedforward':
        model = FeedForwardModel(cfg)
    else:
        model = TwoLayerModel(cfg)


    # Initialize the trainer
    trainer = lightning.Trainer(
        max_epochs=cfg.max_epochs,
        accelerator=cfg.device,
        logger=TensorBoardLogger(cfg.log_dir),
        log_every_n_steps=cfg.log_interval,
    )

    # Train the model
    trainer.fit(model, train_loader, test_loader)

def main():

    parser = argparse.ArgumentParser(description="Aliasing run params")
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to the config file')

    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config, 'r'))

    # Convert the config dictionary to a Config object
    cfg = Config(**cfg)

    # Load the dataset
    train_loader, test_loader = get_mnist_dataloaders(cfg.batch_size, cfg.num_workers)


    for h in range(cfg.min_hidden, cfg.max_hidden+1):
        # Train a single model
        cfg.hidden_dim = h
        cfg.log_dir = f"logs/hidden_{h}"
        train_single_model(cfg, train_loader, test_loader)
        


if __name__ == "__main__":
    main()