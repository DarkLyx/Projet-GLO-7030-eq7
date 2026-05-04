import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
from collections import defaultdict

def plot_model_variants(metrics_dir="results/metrics"):
    """
    Génère des courbes superposant la Validation Loss et la Validation Accuracy 
    pour les différentes variantes (hyperparamètres) d'une même architecture.
    """
    print(" Génération des courbes comparatives intra-architecture...")
    
    history_dir = os.path.join(metrics_dir, "training_history")
    history_files = glob.glob(os.path.join(history_dir, "*_training_history.csv"))
    
    if not history_files:
        print(f" Aucun historique trouvé dans {history_dir}.")
        return

    # Regrouper les fichiers par architecture de base
    architectures = defaultdict(list)
    
    for file in history_files:
        filename = os.path.basename(file)
        full_model_name = filename.replace("_training_history.csv", "")
        base_model_name = full_model_name.split('_OPT')[0]
        
        # On extrait la partie hyperparamètres pour la légende du graphique
        # Si pas d'hyperparamètres (run normal), on met "Default"
        variant_name = full_model_name.replace(f"{base_model_name}_", "") if "_OPT" in full_model_name else "Default"
        
        architectures[base_model_name].append({
            'variant': variant_name,
            'filepath': file
        })

    save_dir = os.path.join(metrics_dir, "comparisons_intra_models")
    os.makedirs(save_dir, exist_ok=True)

    # Pour chaque architecture de base, on crée un double graphique (Loss / Acc)
    for base_model, variants in architectures.items():
        if len(variants) <= 1:
            continue # Inutile de comparer s'il n'y a qu'une seule version
            
        plt.figure(figsize=(15, 6))

        # --- SOUS-GRAPHIQUE 1 : Validation Loss ---
        plt.subplot(1, 2, 1)
        for var in variants:
            df = pd.read_csv(var['filepath'])
            epochs = df['epoch'] if 'epoch' in df.columns else range(1, len(df) + 1)
            plt.plot(epochs, df['val_loss'], marker='.', label=var['variant'])
        
        plt.title(f'Validation Loss - Variantes de {base_model}')
        plt.xlabel('Époques')
        plt.ylabel('Loss')
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.legend()

        # --- SOUS-GRAPHIQUE 2 : Validation Accuracy ---
        plt.subplot(1, 2, 2)
        for var in variants:
            df = pd.read_csv(var['filepath'])
            epochs = df['epoch'] if 'epoch' in df.columns else range(1, len(df) + 1)
            plt.plot(epochs, df['val_acc'], marker='.', label=var['variant'])
            
        plt.title(f'Validation Accuracy - Variantes de {base_model}')
        plt.xlabel('Époques')
        plt.ylabel('Accuracy (%)')
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.legend()

        plt.tight_layout()
        save_path = os.path.join(save_dir, f"compare_variants_{base_model}.png")
        plt.savefig(save_path)
        plt.close()
        print(f" Graphique des variantes pour '{base_model}' généré.")

if __name__ == "__main__":
    plot_model_variants()