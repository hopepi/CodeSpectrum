import torch
import torch.nn as nn
import torch.optim as optim
import model.legacy_resnet_model as resnet # legacy
from config import LR, EPOCHS, DEVICE, Arch
from tqdm import tqdm
from model.arch_multi_model import create_model
from scripts.evaluate import evaluate



def train_model(train_loader, val_loader, arch: Arch, pretrained=True, freeze_backbone=False):
    model = create_model(arch=arch, num_classes=2, pretrained=pretrained, freeze_backbone=freeze_backbone)

    criterion = nn.BCEWithLogitsLoss()  # Binary
    optimizer = optim.Adam(model.parameters(), lr=LR)

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": []
    }
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")
        for images, labels in pbar:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.float().unsqueeze(1).to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            pbar.set_postfix(loss=loss.item())

        # epoch ortalama train loss
        avg_loss = running_loss / len(train_loader.dataset)

        # validation (evaluate imzan device bekliyorsa DEVICE ver)
        val_loss, val_acc = evaluate(model, val_loader, criterion, DEVICE)

        history["train_loss"].append(avg_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch + 1}/{EPOCHS}, "
            f"Train Loss: {avg_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, "
            f"Val Acc: {val_acc * 100:.2f}%"
        )
    return model, history
