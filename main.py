import lightning
import torch
import argparse
from model import FeedForwardModel
import yaml
from config import Config

def train_single_model(cfg):

    # Initialize the model
    model = FeedForwardModel(cfg)


    # Initialize the trainer
    trainer = lightning.Trainer(
        max_epochs=cfg.max_epochs,
        gpus=1 if torch.cuda.is_available() else 0,
        accelerator=cfg.device,
        logger=lightning.loggers.TensorBoardLogger(cfg.log_dir),
        log_every_n_steps=cfg.log_interval,
    )

    # Train the model
    trainer.fit(model)

def main():

    parser = argparse.ArgumentParser(description="Aliasing run params")
    parser.add_argument('--config', type=str, default='configs/config.py', help='Path to the config file')

    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config, 'r'))

    # Convert the config dictionary to a Config object
    cfg = Config(**cfg)

    for _ in range(cfg.max_hidden):
        # Train a single model
        train_single_model(cfg)
        cfg.hidden_dim += 1


if __name__ == "__main__":
    main()