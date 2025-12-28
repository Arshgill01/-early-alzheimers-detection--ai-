"""
Data loading and preprocessing module for Alzheimer's MRI Classification.
Handles dataset loading, augmentation, and train/val/test splitting.
"""

import os
from pathlib import Path
from typing import Tuple, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split


# Class labels mapping
CLASS_NAMES = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {idx: name for name, idx in CLASS_TO_IDX.items()}

# Image configuration
IMAGE_SIZE = 224  # Standard size for pretrained models
MEAN = [0.485, 0.456, 0.406]  # ImageNet normalization
STD = [0.229, 0.224, 0.225]


class AlzheimerMRIDataset(Dataset):
    """Custom Dataset for Alzheimer's MRI images."""
    
    def __init__(
        self, 
        image_paths: List[str], 
        labels: List[int], 
        transform: Optional[transforms.Compose] = None
    ):
        """
        Args:
            image_paths: List of paths to MRI images
            labels: List of corresponding class labels
            transform: Optional transforms to apply
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
            
        return image, label


def get_transforms(is_training: bool = True) -> transforms.Compose:
    """
    Get image transforms for training or validation/testing.
    
    Args:
        is_training: If True, apply data augmentation
        
    Returns:
        Composed transforms
    """
    if is_training:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD)
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD)
        ])


def load_dataset(data_dir: str) -> Tuple[List[str], List[int]]:
    """
    Load all image paths and labels from the dataset directory.
    
    Args:
        data_dir: Root directory containing class subdirectories
        
    Returns:
        Tuple of (image_paths, labels)
    """
    image_paths = []
    labels = []
    
    data_path = Path(data_dir)
    
    for class_name in CLASS_NAMES:
        class_dir = data_path / class_name
        if not class_dir.exists():
            print(f"Warning: {class_dir} does not exist")
            continue
            
        class_idx = CLASS_TO_IDX[class_name]
        
        for img_file in class_dir.iterdir():
            if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                image_paths.append(str(img_file))
                labels.append(class_idx)
                
    return image_paths, labels


def create_data_splits(
    image_paths: List[str], 
    labels: List[int],
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42
) -> Dict[str, Tuple[List[str], List[int]]]:
    """
    Create train/validation/test splits with stratification.
    
    Args:
        image_paths: List of image paths
        labels: List of labels
        test_size: Proportion for test set
        val_size: Proportion for validation set (from remaining after test)
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary with 'train', 'val', 'test' splits
    """
    # First split: separate test set
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        image_paths, labels, 
        test_size=test_size, 
        stratify=labels,
        random_state=random_state
    )
    
    # Second split: separate validation from training
    val_ratio = val_size / (1 - test_size)  # Adjust ratio
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths, train_val_labels,
        test_size=val_ratio,
        stratify=train_val_labels,
        random_state=random_state
    )
    
    return {
        'train': (train_paths, train_labels),
        'val': (val_paths, val_labels),
        'test': (test_paths, test_labels)
    }


def compute_class_weights(labels: List[int]) -> torch.Tensor:
    """
    Compute class weights for handling imbalanced dataset.
    
    Args:
        labels: List of class labels
        
    Returns:
        Tensor of class weights
    """
    class_counts = np.bincount(labels, minlength=len(CLASS_NAMES))
    total = len(labels)
    weights = total / (len(CLASS_NAMES) * class_counts + 1e-6)
    return torch.FloatTensor(weights)


def get_weighted_sampler(labels: List[int]) -> WeightedRandomSampler:
    """
    Create a weighted sampler for handling class imbalance during training.
    
    Args:
        labels: List of class labels
        
    Returns:
        WeightedRandomSampler instance
    """
    class_counts = np.bincount(labels, minlength=len(CLASS_NAMES))
    class_weights = 1.0 / (class_counts + 1e-6)
    sample_weights = [class_weights[label] for label in labels]
    
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(labels),
        replacement=True
    )


def create_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    use_weighted_sampler: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader, torch.Tensor]:
    """
    Create DataLoaders for training, validation, and testing.
    
    Args:
        data_dir: Root directory of the dataset
        batch_size: Batch size for DataLoaders
        num_workers: Number of worker processes
        use_weighted_sampler: Whether to use weighted sampling for training
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader, class_weights)
    """
    # Load dataset
    image_paths, labels = load_dataset(data_dir)
    print(f"Total images found: {len(image_paths)}")
    
    # Create splits
    splits = create_data_splits(image_paths, labels)
    
    # Print split info
    for split_name, (paths, lbls) in splits.items():
        print(f"{split_name}: {len(paths)} images")
        unique, counts = np.unique(lbls, return_counts=True)
        for u, c in zip(unique, counts):
            print(f"  - {IDX_TO_CLASS[u]}: {c}")
    
    # Create datasets
    train_dataset = AlzheimerMRIDataset(
        splits['train'][0], splits['train'][1],
        transform=get_transforms(is_training=True)
    )
    val_dataset = AlzheimerMRIDataset(
        splits['val'][0], splits['val'][1],
        transform=get_transforms(is_training=False)
    )
    test_dataset = AlzheimerMRIDataset(
        splits['test'][0], splits['test'][1],
        transform=get_transforms(is_training=False)
    )
    
    # Compute class weights for loss function
    class_weights = compute_class_weights(splits['train'][1])
    
    # Create DataLoaders
    if use_weighted_sampler:
        train_sampler = get_weighted_sampler(splits['train'][1])
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size,
            sampler=train_sampler, num_workers=num_workers,
            pin_memory=True
        )
    else:
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size,
            shuffle=True, num_workers=num_workers,
            pin_memory=True
        )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, class_weights


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """
    Denormalize a tensor for visualization.
    
    Args:
        tensor: Normalized image tensor
        
    Returns:
        Denormalized numpy array
    """
    mean = np.array(MEAN)
    std = np.array(STD)
    
    image = tensor.cpu().numpy().transpose(1, 2, 0)
    image = std * image + mean
    image = np.clip(image, 0, 1)
    
    return image
