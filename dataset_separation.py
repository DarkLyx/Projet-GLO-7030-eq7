import os
import shutil
import random
import matplotlib.pyplot as plt
import config as cfg

def run_dataset_split():
    source_dir = cfg.RAW_PATH    
    dest_dir = cfg.SPLIT_PATH    
    ratios = {'train': 0.8, 'val': 0.1, 'test': 0.1} 

    if os.path.exists(dest_dir):
        shutil.rmtree(dest_dir)

    print("Début de la répartition des fichiers...")

    for root, dirs, files in os.walk(source_dir):
        if files and not dirs:
            class_name = os.path.basename(root)
            
            random.seed(42)
            random.shuffle(files)
            
            total = len(files)
            train_idx = int(total * ratios['train'])
            val_idx = train_idx + int(total * ratios['val'])
            
            splits_map = {
                'train': files[:train_idx],
                'val': files[train_idx:val_idx],
                'test': files[val_idx:]
            }
            
            for split_name, file_list in splits_map.items():
                target_path = os.path.join(dest_dir, split_name, class_name)
                os.makedirs(target_path, exist_ok=True)
                
                for f in file_list:
                    shutil.copy(os.path.join(root, f), os.path.join(target_path, f))
            
            print(f"Classe '{class_name}' répartie ({total} images)")

    print("\nGénération des graphiques de répartition...")
    show_distribution_charts(dest_dir)

def show_distribution_charts(dest_dir):
    splits = ['train', 'val', 'test']
    colors = ['#ff9999','#66b3ff','#99ff99']
    
    train_path = os.path.join(dest_dir, 'train')
    all_classes = sorted([d for d in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, d))])
    
    n_classes = len(all_classes)
    fig, axes = plt.subplots(1, n_classes, figsize=(22, 6))
    if n_classes == 1: axes = [axes]

    for i, class_name in enumerate(all_classes):
        sizes = []
        for split in splits:
            path = os.path.join(dest_dir, split, class_name)
            count = len(os.listdir(path)) if os.path.exists(path) else 0
            sizes.append(count)
        
        axes[i].pie(sizes, labels=splits, autopct='%1.1f%%', startangle=90, colors=colors)
        axes[i].set_title(f"Classe : {class_name}\n(Total: {sum(sizes)})")

    plt.tight_layout()
    plt.suptitle("Répartition Train/Val/Test pour les 5 classes de cancer", fontsize=16, y=1.05)
    
    os.makedirs(cfg.LOG_DIR, exist_ok=True)
    plt.savefig(os.path.join(cfg.LOG_DIR, "dataset_distribution.png"))
    plt.show()

if __name__ == "__main__":
    run_dataset_split()