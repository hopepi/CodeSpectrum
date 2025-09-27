import os
import json
import torch
import torch.multiprocessing as mp
from datetime import datetime
from config import DEVICE, DATA_DIR, TRAIN_PATH, TEST_PATH, VAL_PATH,Arch
from utils.dataloader import return_data
from scripts.train import train_model
from scripts.test import test_model
from utils.visualize import plot_and_save, save_random_images
from utils.data_inspector import get_image_size_stats, get_class_counts_inform


def main():
    select_model = Arch.VGG16
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("charts", select_model.value, run_id)
    os.makedirs(out_dir, exist_ok=True)

    try:
        size_stats = get_image_size_stats(DATA_DIR, printing=True)
        class_counts, saved_paths = get_class_counts_inform(
            data_dir=DATA_DIR,
            printing=True,
            graph=False,          # ekranda gösterme
            download=True,        # dosyaya kaydet
            out_dir=out_dir
        )
    except Exception as e:
        print(f"Veri okuma sırasında hata: {e}")

    try:
        save_random_images(TRAIN_PATH, save_name="train_random.png", save_dir=out_dir, n=10)
        save_random_images(VAL_PATH,   save_name="val_random.png",   save_dir=out_dir, n=10)
        save_random_images(TEST_PATH,  save_name="test_random.png",  save_dir=out_dir, n=10)
    except Exception as e:
        print(f"Rastgele örnek kaydında hata: {e}")

    train_loader, test_loader, val_loader = return_data()

    model, history = train_model(train_loader, val_loader,Arch.RESNET18)

    test_acc = test_model(model, test_loader, DEVICE, threshold=0.5)

    history["test_acc_percent"] = float(test_acc)

    with open(os.path.join(out_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    title = f"Eğitim Kayıp - {select_model.value}"
    plot_and_save(history["train_loss"],
                  save_name="train_loss.png",
                  save_dir=out_dir,
                  x_label="Epoch",
                  y_label="Loss",
                  title=title)

    plot_and_save(history["val_loss"],
                  save_name="val_loss.png",
                  save_dir=out_dir,
                  x_label="Epoch",
                  y_label="Loss",
                  title=title)

    plot_and_save(history["val_acc"],
                  save_name="val_acc.png",
                  save_dir=out_dir,
                  x_label="Epoch",
                  y_label="Accuracy",
                  title=title)

    os.makedirs("saved_models", exist_ok=True)
    ckpt_path = os.path.join("saved_models", f"{select_model.value}.pth")

    # ImageFolder ise class_to_idx bulunur:
    class_to_idx = getattr(getattr(train_loader, "dataset", None), "class_to_idx", None)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "arch": select_model.value,
            "binary_logits": True,          # BCE + tek logit düzeni
            "test_acc_percent": float(test_acc),  # % doğruluk oranları
            "class_to_idx": class_to_idx,   # benign':0,'malware':1 gibi
        },
        ckpt_path
    )

    print(f"\nTüm işlemler bitti."
          f"\n - Çıktılar: {out_dir}"
          f"\n - Checkpoint: {ckpt_path}")


if __name__ == "__main__":
    mp.freeze_support()  # Windows için gerekli fazla çekirdek önlemi
    main()
