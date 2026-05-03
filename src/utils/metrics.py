import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score, recall_score, f1_score
import os

class ModelEvaluator:
    def __init__(self, class_names, model_name, base_save_dir="results/metrics"):
        self.class_names = class_names
        self.model_name = model_name
        self.base_save_dir = base_save_dir

    def _save_single_metric(self, metric_name, value):
        """Crée un dossier pour la métrique et sauvegarde la valeur en CSV."""
        metric_dir = os.path.join(self.base_save_dir, metric_name)
        os.makedirs(metric_dir, exist_ok=True)
        
        df = pd.DataFrame({'model': [self.model_name], 'value': [value]})
        csv_path = os.path.join(metric_dir, f"{self.model_name}_{metric_name}.csv")
        df.to_csv(csv_path, index=False)

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

        self._save_single_metric("accuracy", acc)
        self._save_single_metric("recall", recall)
        self._save_single_metric("f1_score", f1)
        self._save_single_metric("auc_roc", auc)

        print(f"Précision Globale (Accuracy) : {acc:.4f}")
        print(f"Sensibilité (Recall)         : {recall:.4f}")
        print(f"F1-Score (Pondéré)           : {f1:.4f}")
        print(f"AUC-ROC (OVR)                : {auc:.4f}\n")

      # 1. On calcule la matrice
        cm = confusion_matrix(y_true, y_pred)
        
        # 2. On crée le dossier s'il n'existe pas
        cm_dir = os.path.join(self.base_save_dir, "confusion_matrix")
        os.makedirs(cm_dir, exist_ok=True)
        
        # 3. ON NE FAIT QUE DESSINER ! (Suppression de df_cm.to_csv)
        self._plot_confusion_matrix(cm, cm_dir)
        
    def _plot_confusion_matrix(self, cm, save_dir):
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title(f'Matrice de Confusion - {self.model_name}')
        plt.ylabel('Vraies Étiquettes')
        plt.xlabel('Prédictions du Modèle')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{self.model_name}_cm.png"))
        plt.close()