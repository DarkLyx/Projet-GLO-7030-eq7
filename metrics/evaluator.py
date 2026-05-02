import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score, recall_score, f1_score
import config as cfg
import os

class ModelEvaluator:
    def __init__(self, class_names, save_dir="results/plots", tex_dir="rapport/tab"):
        self.class_names = class_names
        self.save_dir = save_dir
        self.tex_dir = tex_dir
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.tex_dir, exist_ok=True)

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

        report_dict = classification_report(y_true, y_pred, target_names=self.class_names, output_dict=True)
        print(classification_report(y_true, y_pred, target_names=self.class_names))

        cm = confusion_matrix(y_true, y_pred)
        self._plot_confusion_matrix(cm)
        self._export_tex_report(acc, recall, f1, auc, report_dict)

    def _export_tex_report(self, acc, recall, f1, auc, report_dict):
        caption = (f"Rapport de performance --- Approche : \\texttt{{{cfg.APPROACH}}}, "
                   f"Modèle : \\texttt{{{cfg.MODEL_NAME}}}")
        label = f"tab:perf_{cfg.APPROACH}_{cfg.MODEL_NAME}"

        df_global = pd.DataFrame({
            "Métrique": ["Accuracy", "Recall (pondéré)", "F1-Score (pondéré)", "AUC-ROC (OVR)"],
            "Valeur": [acc, recall, f1, auc],
        })

        rows = {c: report_dict[c] for c in self.class_names}
        rows["Macro avg"] = report_dict["macro avg"]
        rows["Weighted avg"] = report_dict["weighted avg"]
        df_class = pd.DataFrame(rows).T[["precision", "recall", "f1-score", "support"]]
        df_class.columns = ["Précision", "Rappel", "F1", "Support"]
        df_class.index.name = "Classe"

        tex_global = df_global.to_latex(index=False, float_format="%.4f", escape=True)
        tex_class = df_class.to_latex(float_format="%.4f", escape=True)

        tex = (
            "\\begin{table}[H]\n"
            "\\centering\n"
            f"\\caption{{{caption}}}\n"
            f"\\label{{{label}}}\n"
            f"{tex_global}\n"
            "\\vspace{0.4cm}\n\n"
            f"{tex_class}"
            "\\end{table}\n"
        )

        tex_path = os.path.join(self.tex_dir, f'perf_report_{cfg.APPROACH}_{cfg.MODEL_NAME}.tex')
        with open(tex_path, 'w', encoding='utf-8') as f:
            f.write(tex)
        print(f"Rapport LaTeX exporté : {tex_path}")

    def _plot_confusion_matrix(self, cm):
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title(f'Matrice de Confusion — {cfg.APPROACH} / {cfg.MODEL_NAME}')
        plt.ylabel('Vraies Étiquettes')
        plt.xlabel('Prédictions')
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, f'confusion_matrix_{cfg.APPROACH}_{cfg.MODEL_NAME}.png')
        plt.savefig(save_path)
        plt.close()