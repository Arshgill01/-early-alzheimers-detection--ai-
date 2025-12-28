"""
Model architectures for Alzheimer's MRI Classification.
Includes custom CNN baseline and EfficientNet transfer learning model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Optional


class ConvBlock(nn.Module):
    """Convolutional block with Conv -> BatchNorm -> ReLU -> Optional MaxPool."""
    
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: int = 3,
        pool: bool = True
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, 
            kernel_size=kernel_size, 
            padding=kernel_size // 2
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2) if pool else nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


class BaselineCNN(nn.Module):
    """
    Custom CNN baseline model for Alzheimer's classification.
    A simple but effective architecture for comparison.
    """
    
    def __init__(self, num_classes: int = 4, dropout: float = 0.5):
        super().__init__()
        
        # Convolutional layers
        self.features = nn.Sequential(
            ConvBlock(3, 32),      # 224 -> 112
            ConvBlock(32, 64),     # 112 -> 56
            ConvBlock(64, 128),    # 56 -> 28
            ConvBlock(128, 256),   # 28 -> 14
            ConvBlock(256, 512),   # 14 -> 7
        )
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(256, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x


class AlzheimerEfficientNet(nn.Module):
    """
    EfficientNet-B0 based model for Alzheimer's classification.
    Uses transfer learning with a custom classification head.
    """
    
    def __init__(
        self, 
        num_classes: int = 4, 
        dropout: float = 0.3,
        pretrained: bool = True,
        freeze_backbone: bool = False
    ):
        super().__init__()
        
        # Load pretrained EfficientNet-B0
        if pretrained:
            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
            self.backbone = models.efficientnet_b0(weights=weights)
        else:
            self.backbone = models.efficientnet_b0(weights=None)
        
        # Get the number of features from the backbone
        num_features = self.backbone.classifier[1].in_features
        
        # Replace the classifier with custom head
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout / 2),
            nn.Linear(512, num_classes)
        )
        
        # Optionally freeze backbone
        if freeze_backbone:
            self._freeze_backbone()
            
    def _freeze_backbone(self):
        """Freeze all backbone layers except the classifier."""
        for name, param in self.backbone.named_parameters():
            if 'classifier' not in name:
                param.requires_grad = False
                
    def unfreeze_backbone(self):
        """Unfreeze all backbone layers for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before the classifier (for Grad-CAM)."""
        return self.backbone.features(x)


class AlzheimerResNet(nn.Module):
    """
    ResNet-50 based model for Alzheimer's classification.
    Alternative transfer learning option.
    """
    
    def __init__(
        self, 
        num_classes: int = 4, 
        dropout: float = 0.3,
        pretrained: bool = True,
        freeze_backbone: bool = False
    ):
        super().__init__()
        
        # Load pretrained ResNet-50
        if pretrained:
            weights = models.ResNet50_Weights.IMAGENET1K_V2
            self.backbone = models.resnet50(weights=weights)
        else:
            self.backbone = models.resnet50(weights=None)
        
        # Get the number of features
        num_features = self.backbone.fc.in_features
        
        # Replace final fully connected layer
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout / 2),
            nn.Linear(512, num_classes)
        )
        
        if freeze_backbone:
            self._freeze_backbone()
            
    def _freeze_backbone(self):
        """Freeze all backbone layers except fc."""
        for name, param in self.backbone.named_parameters():
            if 'fc' not in name:
                param.requires_grad = False
                
    def unfreeze_backbone(self):
        """Unfreeze all backbone layers."""
        for param in self.backbone.parameters():
            param.requires_grad = True
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def create_model(
    model_name: str = 'efficientnet',
    num_classes: int = 4,
    pretrained: bool = True,
    dropout: float = 0.3,
    freeze_backbone: bool = False
) -> nn.Module:
    """
    Factory function to create models.
    
    Args:
        model_name: One of 'baseline', 'efficientnet', 'resnet'
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        dropout: Dropout rate
        freeze_backbone: Whether to freeze backbone initially
        
    Returns:
        Model instance
    """
    model_name = model_name.lower()
    
    if model_name == 'baseline':
        return BaselineCNN(num_classes=num_classes, dropout=dropout)
    elif model_name == 'efficientnet':
        return AlzheimerEfficientNet(
            num_classes=num_classes,
            dropout=dropout,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone
        )
    elif model_name == 'resnet':
        return AlzheimerResNet(
            num_classes=num_classes,
            dropout=dropout,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone
        )
    else:
        raise ValueError(f"Unknown model: {model_name}. Choose from 'baseline', 'efficientnet', 'resnet'")


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_summary(model: nn.Module, input_size: tuple = (1, 3, 224, 224)) -> str:
    """Get a summary of the model architecture."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = count_parameters(model)
    
    summary = f"""
Model Summary:
{'='*50}
Total Parameters: {total_params:,}
Trainable Parameters: {trainable_params:,}
Non-trainable Parameters: {total_params - trainable_params:,}
{'='*50}
"""
    return summary
