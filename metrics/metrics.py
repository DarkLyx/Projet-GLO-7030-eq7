import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import os

class ModelEvaluator:
    def __init__(self, class_names, save_dir="results/plots"):
        self.class_names = class_names
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def evaluate_predictions(self, y_true, y_pred):
        print("\n" + "="*50)
        print("RAPPORT DE CLASSIFICATION")
        print("="*50)
        report = classification_report(y_true, y_pred, target_names=self.class_names)
        print(report)

        cm = confusion_matrix(y_true, y_pred)
        self._plot_confusion_matrix(cm)

    def _plot_confusion_matrix(self, cm):
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=self.class_names, yticklabels=self.class_names)
        
        plt.title('Matrice de Confusion')
        plt.ylabel('Vraies Étiquettes')
        plt.xlabel('Prédictions du Modèle')
        plt.tight_layout()
        
        save_path = os.path.join(self.save_dir, 'confusion_matrix.png')
        plt.savefig(save_path)
        plt.close()
        print(f"Matrice de confusion sauvegardée sous : {save_path}")