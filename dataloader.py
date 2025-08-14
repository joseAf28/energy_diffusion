import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def get_cifar10_dataloaders(batch_size=64, data_dir='./data'):
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )

    # Do the same for the test data.
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    )
    
    print("Dataset downloaded and loaded.")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,       # Shuffle the training data each epoch
        num_workers=4,      # Use multiple subprocesses to load data
        pin_memory=True     # Speeds up data transfer to the GPU
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,      # No need to shuffle test data
        num_workers=4,
        pin_memory=True
    )

    print(f"DataLoaders created with batch size: {batch_size}")
    return train_loader, test_loader


if __name__ == '__main__':
    BATCH_SIZE = 64
    
    # Get the dataloaders
    train_loader, test_loader = get_cifar10_dataloaders(batch_size=BATCH_SIZE)
    
    # Let's inspect a single batch to verify
    print("\n--- Inspecting a batch ---")
    
    # Get one batch of training images
    images, labels = next(iter(train_loader))
    
    # Print the shape of the tensors
    print(f"Images tensor shape: {images.shape}") # Should be [batch_size, 3, 32, 32]
    print(f"Labels tensor shape: {labels.shape}") # Should be [batch_size]
    
    # Verify the pixel value range (should be approx. [-1, 1])
    print(f"Image tensor min value: {images.min():.2f}")
    print(f"Image tensor max value: {images.max():.2f}")