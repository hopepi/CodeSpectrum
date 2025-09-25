import torch
import torch.nn as nn
import torch.optim as optim
import model.resnet_model as resnet
from config import LR, EPOCHS, DEVICE
from tqdm import tqdm
import torch.multiprocessing as mp
from scripts.evaluate import evaluate
from utils.dataloader import return_data



def train_model():
    model = resnet.create_model(num_classes=2, pretrained=True, freeze_backbone=False)
    criterion = nn.BCEWithLogitsLoss()  # Binary
    optimizer = optim.Adam(model.parameters(), lr=LR)

    train_loader, _, val_loader = return_data()

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

        print(
            f"Epoch {epoch + 1}/{EPOCHS}, "
            f"Train Loss: {avg_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, "
            f"Val Acc: {val_acc * 100:.2f}%"
        )


if __name__ == "__main__":
    mp.freeze_support()  # Windows  multiproc için lazım
    train_model()
