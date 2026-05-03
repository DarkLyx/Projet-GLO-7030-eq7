import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob

def generate_comparison_barchart(metrics_dir="results/metrics"):
    print(" Génération du graphique de comparaison global (Meilleures versions uniquement)...")
    
    history_dir = os.path.join(metrics_dir, "training_history")
    accuracy_dir = os.path.join(metrics_dir, "accuracy")
    test_acc_files = glob.glob(os.path.join(accuracy_dir, "*_accuracy.csv"))
    
    if not test_acc_files:
        print(f" Aucun fichier de Test Accuracy trouvé dans {accuracy_dir}.")
        return

    best_models = {}

    for test_file in test_acc_files:
        filename = os.path.basename(test_file)
        full_model_name = filename.replace("_accuracy.csv", "")
        
        base_model_name = full_model_name.split('_LR')[0]
        
        df_test = pd.read_csv(test_file)
        test_acc = df_test['value'].iloc[0] * 100
        
        history_file = os.path.join(history_dir, f"{full_model_name}_training_history.csv")
        if os.path.exists(history_file):
            df_hist = pd.read_csv(history_file)
            val_acc = df_hist['val_acc'].max()
        else:
            val_acc = 0.0

        # Si cette architecture n'est pas encore enregistrée OU si cette version est meilleure
        if base_model_name not in best_models or val_acc > best_models[base_model_name]['val_acc']:
            best_models[base_model_name] = {
                'full_name': full_model_name,
                'val_acc': val_acc,
                'test_acc': test_acc
            }

    labels = []
    val_accuracies = []
    test_accuracies = []

    for base_name, data in best_models.items():
        labels.append(data['full_name'])
        val_accuracies.append(data['val_acc'])
        test_accuracies.append(data['test_acc'])

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 7))
    rects1 = ax.bar(x - width/2, val_accuracies, width, label='Best Validation Accuracy', color='#ff9999', edgecolor='black')
    rects2 = ax.bar(x + width/2, test_accuracies, width, label='Test Accuracy (Final)', color='#66b3ff', edgecolor='black')

    ax.set_ylabel('Précision / Accuracy (%)')
    ax.set_title('Comparaison des Meilleurs Modèles par Architecture')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend(loc='lower right')
    ax.set_ylim(0, 110)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            if height > 0:
                ax.annotate(f'{height:.1f}%', xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()
    save_path = os.path.join(metrics_dir, "comparaison_globale_meilleurs_modeles.png")
    plt.savefig(save_path)
    plt.close()
    print(f" Graphique global sauvegardé sous : {save_path}")