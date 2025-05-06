from dataclasses import dataclass
import torch

@dataclass
class Config:
    """Configuration for the FeedForward model."""
    # Model Type
    model_type: str = 'feedforward'

    # Model parameters
    num_layers: int = 3
    input_dim: int = 128
    hidden_dim: int = 0
    output_dim: int = 10
    activation: str = 'relu'

    # Training parameters
    activation: str = 'relu'
    loss_fn: str = 'mse'
    optimizer: str = 'sgd'
    learning_rate: float = 0.001
    max_epochs: int = 10

    min_hidden: int = 1
    max_hidden: int = 100

    # Data parameters
    batch_size: int = 32
    num_workers: int = 4

    # Logging parameters
    log_dir: str = 'logs'
    log_interval: int = 10
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataset: str = 'mnist'
