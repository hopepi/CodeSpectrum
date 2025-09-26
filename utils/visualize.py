import os
import random
import matplotlib.pyplot as plt
from PIL import Image


def plot_and_save(values,
                  save_name,
                  save_dir="charts",
                  x_label="Epoch",
                  y_label="Value",
                  title="Grafik"):

    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(6, 4))
    plt.plot(values, marker="o")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()

    save_path = os.path.join(save_dir, save_name)
    plt.savefig(save_path, dpi=200)
    plt.close()

    print(f"Dosya yoluna : {save_path}, Kaydedildi")


def save_random_images(data_dir, save_name="random_samples.png", save_dir="charts", n=10):

    os.makedirs(save_dir, exist_ok=True)

    # Tüm resim yollarını topla
    image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".gif")
    all_images = []
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.lower().endswith(image_extensions):
                all_images.append(os.path.join(root, f))

    if len(all_images) == 0:
        print(f"{data_dir} içinde uygun görsel bulunamadı.")
        return None

    sample_paths = random.sample(all_images, min(n, len(all_images)))

    cols = min(5, n)
    rows = (len(sample_paths) + cols - 1) // cols

    plt.figure(figsize=(cols * 3, rows * 3))

    for i, img_path in enumerate(sample_paths, 1):
        try:
            img = Image.open(img_path)
            plt.subplot(rows, cols, i)
            plt.imshow(img)
            plt.axis("off")
            plt.title(os.path.basename(os.path.dirname(img_path)))  # sınıf ismi
        except Exception:
            continue

    plt.tight_layout()
    save_path = os.path.join(save_dir, save_name)
    plt.savefig(save_path, dpi=200)
    plt.close()

    print(f"Rastgele örnek görseller kaydedildi: {save_path}")
    return save_path