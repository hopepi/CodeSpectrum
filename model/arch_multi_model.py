import torch.nn as nn
from torchvision import models
from config import DEVICE, Arch


def create_model(arch: Arch, num_classes=2, pretrained=True, freeze_backbone=False):
    if arch == Arch.RESNET18:
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        num_ftrs = model.fc.in_features
        head_attr = "fc"

    elif arch == Arch.MOBILENET_V3_SMALL:
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        num_ftrs = model.classifier[0].in_features
        head_attr = "classifier"

    elif arch == Arch.EFFICIENTNET_B0:
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        num_ftrs = model.classifier[1].in_features
        head_attr = "classifier"

    elif arch == Arch.DENSENET121:
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        model = models.densenet121(weights=weights)
        num_ftrs = model.classifier.in_features
        head_attr = "classifier"

    elif arch == Arch.VGG16:
        weights = models.VGG16_Weights.DEFAULT if pretrained else None
        model = models.vgg16(weights=weights)
        num_ftrs = model.classifier[6].in_features
        head_attr = "classifier"

    elif arch == Arch.MOBILENET_V2:
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v2(weights=weights)
        num_ftrs = model.classifier[1].in_features
        head_attr = "classifier"

    elif arch == Arch.SHUFFLENET_V2_X0_5:
        weights = models.ShuffleNet_V2_X0_5_Weights.DEFAULT if pretrained else None
        model = models.shufflenet_v2_x0_5(weights=weights)
        num_ftrs = model.fc.in_features
        head_attr = "fc"

    elif arch == Arch.MNASNET_1_0:
        weights = models.MNASNet1_0_Weights.DEFAULT if pretrained else None
        model = models.mnasnet1_0(weights=weights)
        num_ftrs = model.classifier[1].in_features
        head_attr = "classifier"

    else:
        raise ValueError(f"Desteklenmeyen model : {arch}")

    if freeze_backbone:
        for name, param in model.named_parameters():
            if not name.startswith(f"{head_attr}"):
                param.requires_grad = False

    if num_classes == 2:
        new_head = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )
    else:
        new_head = nn.Sequential(
            nn.Linear(num_ftrs, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    setattr(model, head_attr, new_head)
    return model.to(DEVICE)
