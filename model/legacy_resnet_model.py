import torch
import torch.nn as nn
from torchvision import models
from config import DEVICE


def create_model(num_classes = 2, pretrained = True, freeze_backbone=False):

    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    num_ftrs = model.fc.in_features

    if num_classes == 2:
        """
        sigmoid çıkış
        """
        model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256,1)# tek nöron
        )
    else:
        # softmax kullanılır
        model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    return model.to(device=DEVICE)