# Fichier : utils/visualizer.py

import matplotlib.pyplot as plt
import config as cfg
import os

def plot_training_history(train_losses, val_losses, train_accs, val_accs, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.plot(epochs, train_losses, 'b-', label='Train Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
    ax1.set_title('Évolution de la Perte (Loss)')
    ax1.set_xlabel('Époques')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs, train_accs, 'b-', label='Train Accuracy')
    ax2.plot(epochs, val_accs, 'r-', label='Validation Accuracy')
    ax2.set_title('Évolution de la Précision (Accuracy)')
    ax2.set_xlabel('Époques')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    save_path = os.path.join(save_dir, f'training_history_{cfg.APPROACH}_{cfg.MODEL_NAME}.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Graphique d'entraînement sauvegardé sous : {save_path}")