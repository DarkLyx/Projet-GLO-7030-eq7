import matplotlib.pyplot as plt
import pandas as pd
import os

def plot_training_history(history, save_dir, model_name):
    """
    history: dict contenant 'train_loss', 'val_loss', 'train_acc', 'val_acc'
    """
    os.makedirs(save_dir, exist_ok=True)
    
    df = pd.DataFrame(history)
    df.index += 1  
    df.index.name = "epoch"
    
    csv_path = os.path.join(save_dir, f"{model_name}_training_history.csv")
    df.to_csv(csv_path)
    print(f"Données d'entraînement sauvegardées sous : {csv_path}")

    epochs = range(1, len(df) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, df['train_loss'], color='blue', marker='o', linestyle='-', label='Entraînement (Train)')
    plt.plot(epochs, df['val_loss'], color='red', marker='s', linestyle='--', label='Validation (Val)')
    
    plt.title(f'Évolution de la Perte (Loss) - {model_name}')
    plt.xlabel('Époques')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    
    loss_path = os.path.join(save_dir, f"{model_name}_loss.png")
    plt.savefig(loss_path)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, df['train_acc'], color='blue', marker='o', linestyle='-', label='Entraînement (Train)')
    plt.plot(epochs, df['val_acc'], color='red', marker='s', linestyle='--', label='Validation (Val)')
    
    plt.title(f'Évolution de la Précision (Accuracy) - {model_name}')
    plt.xlabel('Époques')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    
    acc_path = os.path.join(save_dir, f"{model_name}_accuracy.png")
    plt.savefig(acc_path)
    plt.close()
            
    print(f"Graphiques Loss et Accuracy combinés sauvegardés dans : {save_dir}")