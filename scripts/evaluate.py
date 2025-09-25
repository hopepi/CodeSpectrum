import torch
from config import DEVICE


@torch.no_grad()
def evaluate(model, loader, criterion, device, threshold=0.5):
    model.eval()
    total_loss, n = 0.0, 0
    correct = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.float().unsqueeze(1).to(device, non_blocking=True)

        logits = model(images)
        loss   = criterion(logits, labels)

        bs = images.size(0)
        total_loss += loss.item() * bs
        n += bs

        probs = torch.sigmoid(logits)
        preds = (probs >= threshold).float()
        correct += (preds == labels).sum().item()

    avg_loss = total_loss / max(n, 1)
    acc = correct / max(n, 1)
    return avg_loss, acc
