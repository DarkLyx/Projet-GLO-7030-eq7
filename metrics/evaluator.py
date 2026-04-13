import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score, recall_score, f1_score
import os

class ModelEvaluator:
    def __init__(self, class_names, save_dir="results/plots"):
        self.class_names = class_names
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def evaluate_predictions(self, y_true, y_pred, y_probs):
        print("\n" + "="*50)
        print("RAPPORT DE PERFORMANCE (MÉTRIQUES)")
        print("="*50)

        acc = accuracy_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')

        try:
            auc = roc_auc_score(y_true, y_probs, multi_class='ovr', average='weighted')
        except ValueError:
            auc = float('nan')
            print("Impossible de calculer l'AUC (classes manquantes dans le test).")

        print(f" récision Globale (Accuracy) : {acc:.4f}")
        print(f"Sensibilité (Recall)         : {recall:.4f}")
        print(f"F1-Score (Pondéré)          : {f1:.4f}")
        print(f"AUC-ROC (OVR)                : {auc:.4f}\n")

        print(classification_report(y_true, y_pred, target_names=self.class_names))

        cm = confusion_matrix(y_true, y_pred)
        self._plot_confusion_matrix(cm)

    def _plot_confusion_matrix(self, cm):
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title('Matrice de Confusion')
        plt.ylabel('Vraies Étiquettes')
        plt.xlabel('Prédictions')
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, 'confusion_matrix.png')
        plt.savefig(save_path)
        plt.close()