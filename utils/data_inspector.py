import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from config import DATA_DIR

def _plot_counts(counts_dict, title="Class Distribution", save_path=None, show=False):
    """
    Tek bir split için {class: count} sözlüğünü çubuk grafik olarak çizer.
    show=True ise ekranda gösterir.
    save_path verilirse diske kaydeder.
    """
    if not counts_dict:  # boş sözlük -> iş yok
        return None

    classes = list(counts_dict.keys())
    values  = list(counts_dict.values())

    plt.figure(figsize=(6, 4))
    plt.bar(classes, values)
    plt.title(title)
    plt.xlabel("Sınıf")
    plt.ylabel("Adet")

    # bar üstüne sayı yaz
    for i, v in enumerate(values):
        plt.text(i, v, str(v), ha="center", va="bottom", fontsize=9)

    plt.tight_layout()

    # Kaydet
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    # Göster / kapat
    if show:
        plt.show()
    else:
        plt.close()

    return save_path


def get_class_counts_inform(data_dir, printing=True, graph=False, download=False, out_dir="charts"):
    """
    train test val splitlerini bulur
    Her split için sınıf başına görsel sayar
    graph: ekranda gösterilsin mi
    download: diske kaydedilsin mi
    Dönüş: all_counts, saved_paths
    """
    splits = []
    for s in ["train", "test", "val"]:
        p = os.path.join(data_dir, s)
        if os.path.isdir(p):
            splits.append(s)

    if "train" not in splits or "test" not in splits:
        raise ValueError("train veya test bulunamadı, data_dir kontrol et")


    print("Bulunan split'ler:", splits) if printing else None

    image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".gif")
    all_counts = {}
    saved_paths = {}

    for s in splits:
        split_dir = os.path.join(data_dir, s)

        print(f"\nSplit: {s}") if printing else None

        # Sınıfları bul
        entries = os.listdir(split_dir)
        classes = [name for name in entries if os.path.isdir(os.path.join(split_dir, name))]

        print("  Sınıflar:", classes) if printing else None

        # Sayım
        counts = {}
        for cls in classes:
            folder = os.path.join(split_dir, cls)
            n = sum(
                1
                for fname in os.listdir(folder)
                if os.path.isfile(os.path.join(folder, fname))
                and fname.lower().endswith(image_extensions)
            )
            counts[cls] = n

            print(f"  - {cls}: {n} görsel") if printing else None
        all_counts[s] = counts

        make_plot = graph or download
        if make_plot:
            save_path = os.path.join(out_dir, f"{s}_counts.png") if download else None
            path = _plot_counts(
                counts,
                title=f"{s} — Class Distribution",
                save_path=save_path,
                show=graph  # sadece graph True ise göster
            )
            saved_paths[s] = path  # download=False ise None kalır


    print("\nÖzet:", all_counts) if printing else None

    return all_counts, saved_paths


def get_image_size_stats(data_dir, printing=True):
    splits = []
    for s in ["train", "test", "val"]:
        p = os.path.join(data_dir, s)
        if os.path.isdir(p):
            splits.append(s)
    print("Bulunan split'ler:", splits) if printing else None

    stats = {}

    for split in splits:
        split_dir = os.path.join(data_dir, split)
        widths, heights = [], []

        for cls in os.listdir(split_dir):
            cdir = os.path.join(split_dir, cls)
            if not os.path.isdir(cdir):
                continue

            for f in os.listdir(cdir):
                path = os.path.join(cdir, f)
                if not os.path.isfile(path):
                    continue
                try:
                    with Image.open(path) as img:
                        w, h = img.size
                        widths.append(w)
                        heights.append(h)
                except Exception:
                    pass

        if not widths:
            print(f"{split}: ölçülecek görsel bulunamadı") if printing else None
            continue

        arr_w, arr_h = np.array(widths), np.array(heights)
        stat = {
            "data_type": split,
            "count": int(len(arr_w)),
            "width_min": int(arr_w.min()),
            "width_max": int(arr_w.max()),
            "width_mean": float(arr_w.mean()),
            "height_min": int(arr_h.min()),
            "height_max": int(arr_h.max()),
            "height_mean": float(arr_h.mean()),
            "aspect_ratio_mean": float((arr_w / arr_h).mean())
        }


        stats[split] = stat
        print(f"{split}: {stat}") if printing else None

    print("Görsel boyut istatistikleri özet:", stats) if printing else None
    return stats

get_image_size_stats(DATA_DIR, printing=True)
get_class_counts_inform(data_dir=DATA_DIR,printing=True,graph=False,download=True,out_dir="../charts")