import os
import csv
import shutil
import random
import matplotlib.pyplot as plt
import configs.config as cfg

CLEAN_REPO = "external/LC25000-clean"
ANNOT_TEMPLATE = os.path.join(
    CLEAN_REPO, "annotations", "{class_name}", "UNI", "resize_only", "final_clusters.csv"
)
CLASS_DIRS = {
    "colon_aca": ("colon_image_sets", "colon_aca"),
    "colon_n":   ("colon_image_sets", "colon_n"),
    "lung_aca":  ("lung_image_sets",  "lung_aca"),
    "lung_n":    ("lung_image_sets",  "lung_n"),
    "lung_scc":  ("lung_image_sets",  "lung_scc"),
}


def load_clusters(class_name):
    """Return {cluster_label: [filename, ...]} from the LC25000-clean annotations."""
    path = ANNOT_TEMPLATE.format(class_name=class_name)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Annotations introuvables : {path}. "
            "Vérifie que le sous-module external/LC25000-clean est bien initialisé "
            "(`git submodule update --init --recursive`)."
        )
    clusters = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = int(row["cluster_label"])
            fname = os.path.basename(row["img_path"])
            clusters.setdefault(label, []).append(fname)
    return clusters


def split_clusters(cluster_ids, ratios, rng):
    """Split a list of cluster IDs into train/val/test according to ratios."""
    ids = list(cluster_ids)
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(n * ratios["train"])
    n_val = int(n * ratios["val"])
    return {
        "train": set(ids[:n_train]),
        "val":   set(ids[n_train:n_train + n_val]),
        "test":  set(ids[n_train + n_val:]),
    }


def run_dataset_split():
    source_dir = cfg.RAW_PATH
    dest_dir = cfg.SPLIT_PATH
    ratios = {"train": 0.8, "val": 0.1, "test": 0.1}

    if os.path.exists(dest_dir):
        shutil.rmtree(dest_dir)

    print("Début de la répartition par cluster (LC25000-clean)...")
    rng = random.Random(42)

    for class_name, (subset, cls_dir) in CLASS_DIRS.items():
        clusters = load_clusters(class_name)
        assignment = split_clusters(clusters.keys(), ratios, rng)

        src_class_dir = os.path.join(source_dir, "lung_colon_image_set", subset, cls_dir)
        counts = {"train": 0, "val": 0, "test": 0}

        for split_name, cluster_ids in assignment.items():
            target_path = os.path.join(dest_dir, split_name, class_name)
            os.makedirs(target_path, exist_ok=True)
            for cid in cluster_ids:
                for fname in clusters[cid]:
                    src = os.path.join(src_class_dir, fname)
                    if not os.path.exists(src):
                        continue
                    shutil.copy(src, os.path.join(target_path, fname))
                    counts[split_name] += 1

        n_clusters = len(clusters)
        print(
            f"Classe '{class_name}': {n_clusters} clusters → "
            f"train={counts['train']}, val={counts['val']}, test={counts['test']}"
        )

    print("\nGénération des graphiques de répartition...")
    show_distribution_charts(dest_dir)


def show_distribution_charts(dest_dir):
    splits = ['train', 'val', 'test']
    colors = ['#ff9999', '#66b3ff', '#99ff99']

    train_path = os.path.join(dest_dir, 'train')
    all_classes = sorted([d for d in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, d))])

    n_classes = len(all_classes)
    fig, axes = plt.subplots(1, n_classes, figsize=(22, 6))
    if n_classes == 1:
        axes = [axes]

    for i, class_name in enumerate(all_classes):
        sizes = []
        for split in splits:
            path = os.path.join(dest_dir, split, class_name)
            count = len(os.listdir(path)) if os.path.exists(path) else 0
            sizes.append(count)

        axes[i].pie(sizes, labels=splits, autopct='%1.1f%%', startangle=90, colors=colors)
        axes[i].set_title(f"Classe : {class_name}\n(Total: {sum(sizes)})")

    plt.tight_layout()
    plt.suptitle("Répartition Train/Val/Test par cluster (LC25000-clean)", fontsize=16, y=1.05)

    os.makedirs(cfg.LOG_DIR, exist_ok=True)
    plt.savefig(os.path.join(cfg.LOG_DIR, "dataset_distribution.png"))
    plt.show()


if __name__ == "__main__":
    run_dataset_split()