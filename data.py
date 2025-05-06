import torchvision
from torchvision import transforms
from torch.utils.data import ConcatDataset
import torch
import os

def get_mnist_dataloaders(batch_size, num_workers):
    # Define the transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: torch.flatten(x))  # Flatten the images
    ])

    # Load the MNIST dataset with the transformations
    train_dataset = torchvision.datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = torchvision.datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    # Compute the risk on the train and test datasets
    test_dataset = ConcatDataset([train_dataset, test_dataset])

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, test_loader
