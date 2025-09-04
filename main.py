import torch
import torchvision.transforms as transforms
from torch.utils.data import random_split
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from utils.MACROS import NUM_WORKERS, BATCH_SIZE
from utils.show_cifar_image import show_tensor_image
import numpy as np
import pandas as pd


if __name__ == "__main__":

    transform_train = transforms.Compose([
        transforms.ToTensor(),                        # Conversión a tensor
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # Normalización
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])


    trainset = CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    train_size = int(0.8 * len(trainset))
    val_size = len(trainset) - train_size

    torch.manual_seed(42)
    trainset, valset = random_split(trainset, [train_size, val_size])
    valset.dataset.transform = transform_test

    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    valloader = DataLoader(valset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)



