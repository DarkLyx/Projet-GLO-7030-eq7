import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def generate_comparison_barchart(metrics_dir="results/metrics"):
    print(" Génération du graphique de comparaison global (Champions)...")
    
    summary_path = os.path.join(metrics_dir, "champions_summary.csv")
    
    if not os.path.exists(summary_path):
        print(f" Aucun fichier {summary_path} trouvé. Lancez d'abord le mode 'eval' pour élire les champions.")
        return

    # On lit simplement le registre !
    df = pd.read_csv(summary_path)
    
    labels = df['Variant'].tolist()
    val_accuracies = df['Val_Accuracy'].tolist()
    test_accuracies = df['Test_Accuracy'].tolist()

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 7))
    rects1 = ax.bar(x - width/2, val_accuracies, width, label='Best Validation Accuracy', color='#ff9999', edgecolor='black')
    rects2 = ax.bar(x + width/2, test_accuracies, width, label='Test Accuracy (Final)', color='#66b3ff', edgecolor='black')

    ax.set_ylabel('Précision / Accuracy (%)')
    ax.set_title('Comparaison des Modèles Champions (Val + Test)')
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
    save_path = os.path.join(metrics_dir, "comparaison_globale_champions.png")
    plt.savefig(save_path)
    plt.close()
    print(f" Graphique global sauvegardé sous : {save_path}")