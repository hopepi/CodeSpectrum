import torch

@torch.no_grad()
def test_model(model, test_loader, device, threshold=0.5):
    model.eval()
    correct, total = 0, 0

    for images, labels in test_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.float().unsqueeze(1).to(device, non_blocking=True)  # [B,1]

        logits = model(images)                   # [B,1]
        probs  = torch.sigmoid(logits)           # [B,1]
        preds  = (probs >= threshold).float()    # [B,1]

        total   += labels.size(0)
        correct += (preds == labels).sum().item()

    acc = 100.0 * correct / max(total, 1)
    print(f"[TEST] Accuracy: {acc:.2f}%")
    return acc