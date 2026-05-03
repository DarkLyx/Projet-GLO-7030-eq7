import configs.config as cfg

from src.data_processing.download_dataset import download_from_kaggle
from src.data_processing.preprocessing import run_preprocessing
from src.engine.train import run_training
from src.engine.evaluate import run_evaluation
from src.data_processing.dataset_separation import run_dataset_split
from src.engine.grid_search import run_grid_search
from src.utils.compare import generate_comparison_barchart
from src.utils.compare_modeltype import plot_model_variants
import os
import pandas as pd
import glob

def main():
    print("=" * 60)
    print("CANCER CLASSIFICATION PROJECT LAUNCH")
    print(f"CURRENT MODE     : {cfg.MODE}")
    print(f"CURRENT APPROACH : {cfg.APPROACH}")
    print("=" * 60)

    if cfg.MODE == "download":
        print(">> Running Kaggle dataset download...")
        download_from_kaggle()

    elif cfg.MODE == "preprocess":
        print(">> Running dataset split...")
        run_dataset_split()
        print(">> Running data preprocessing (CLAHE, Resizing)...")
        run_preprocessing() 

    elif cfg.MODE == "train":
        if cfg.APPROACH == "baseline":
            print(f">> Running Convolutional Neural Network ({cfg.MODEL_NAME}) approach...")
            run_training()

        elif cfg.APPROACH == "transformers":
            print(f">> Running Vision Transformer ({cfg.MODEL_NAME}) approach...")
            run_training()

        elif cfg.APPROACH == "hybrids":
            print(f">> Running Hybrids Models ({cfg.MODEL_NAME}) approach...")
            run_training()
            
        elif cfg.APPROACH == "kan":
            print(">> Running Vision KAN (Kolmogorov-Arnold Networks) approach...")
            run_training()
            
        elif cfg.APPROACH == "ssm":
            print(">> Running State Space Models (Mamba) approach...")
            run_training()
            
        else:
            print(f"ERROR: The approach '{cfg.APPROACH}' is unknown for training.")
            print("Please use 'baseline', 'transformers', 'hybrids', 'kan', or 'ssm'.")

    elif cfg.MODE == "eval":
        for i, exp in enumerate(cfg.EXPERIMENTS_QUEUE):
            cfg.APPROACH = exp["approach"]
            cfg.MODEL_NAME = exp["model"]
            
            print(f"\n" + "="*60)
            print(f" TOURNOI GLOBAL (VAL + TEST) : {cfg.MODEL_NAME} ({i+1}/{len(cfg.EXPERIMENTS_QUEUE)})")
            print("="*60)
            
            # 1. On cherche toutes les variantes .pth de ce modèle
            pattern = os.path.join(cfg.CHECKPOINT_DIR, f"best_{cfg.MODEL_NAME}*.pth")
            model_files = glob.glob(pattern)
            
            # On exclut le fichier final canonique (ex: best_resnet18.pth) s'il existe déjà
            nom_canonique = f"best_{cfg.MODEL_NAME}.pth"
            model_files = [f for f in model_files if os.path.basename(f) != nom_canonique]

            if not model_files:
                print(f" Aucun checkpoint variante trouvé pour {cfg.MODEL_NAME}.")
                continue
                
            best_run_name = None
            best_combined_score = -1.0 # Le score maximum sera sur 200 (100% Val + 100% Test)
            best_pth_path = None
            best_val_score = 0.0
            best_test_score = 0.0
            
            dossier_historique = os.path.join("results", "metrics", "training_history")
            
            # 2. Tournoi : On évalue chaque variante
            for file_path in model_files:
                filename = os.path.basename(file_path)
                run_name = filename.replace("best_", "").replace(".pth", "")
                
                print(f"\n>> Analyse du candidat : {run_name}")
                
                val_acc_percent = 0.0
                csv_path = os.path.join(dossier_historique, f"{run_name}_training_history.csv")
                try:
                    if os.path.exists(csv_path):
                        df = pd.read_csv(csv_path)
                        if 'val_acc' in df.columns:
                            val_acc_percent = df['val_acc'].max()
                except Exception as e:
                    print(f" Erreur CSV : {e}")
                    
                # --- B. Score de Test (En lançant l'évaluation) ---
                test_acc_raw = run_evaluation(run_name=run_name)
                
                if test_acc_raw is None:
                    continue
                    
                # On le met en pourcentage pour l'additionner à la Validation
                test_acc_percent = test_acc_raw * 100.0
                
                # --- C. Le Score Combiné ---
                combined_score = val_acc_percent + test_acc_percent
                
                print(f" Val: {val_acc_percent:.2f}% | Test: {test_acc_percent:.2f}% | Total: {combined_score:.2f} / 200")
                
                # Si ce candidat bat le record, il devient le nouveau roi
                if combined_score > best_combined_score:
                    best_combined_score = combined_score
                    best_run_name = run_name
                    best_pth_path = file_path
                    best_val_score = val_acc_percent
                    best_test_score = test_acc_percent
                    
            # 3. Couronnement et copie binaire
            if best_run_name and best_pth_path:
                print("\n" + "-" * 15)
                print(f"LE GAGNANT ABSOLU (VAL+TEST) EST :")
                print(f"   {best_run_name}")
                print(f"SCORE TOTAL : {best_combined_score:.2f} / 200")
                print("-" * 15 + "\n")
                
                destination = os.path.join(cfg.CHECKPOINT_DIR, nom_canonique)
                
                try:
                    with open(best_pth_path, 'rb') as f_source:
                        with open(destination, 'wb') as f_dest:
                            while chunk := f_source.read(8192):
                                f_dest.write(chunk)
                    print(f"✅ Modèle sauvegardé avec succès et prêt pour la comparaison finale !")
                except Exception as e:
                    print(f"❌ Erreur lors de la copie binaire : {e}")
                
                summary_path = os.path.join("results", "metrics", "champions_summary.csv")
                champ_df = pd.DataFrame({
                    'Architecture': [cfg.MODEL_NAME],
                    'Variant': [best_run_name],
                    'Val_Accuracy': [best_val_score],
                    'Test_Accuracy': [best_test_score]
                })
                
                try:
                    # On ajoute ou on met à jour le registre
                    if os.path.exists(summary_path):
                        df_existing = pd.read_csv(summary_path)
                        # On supprime l'ancien champion de cette architecture s'il y en avait un
                        df_existing = df_existing[df_existing['Architecture'] != cfg.MODEL_NAME]
                        df_final = pd.concat([df_existing, champ_df], ignore_index=True)
                    else:
                        df_final = champ_df
                        
                    df_final.to_csv(summary_path, index=False)
                    print(f"Statistiques officielles enregistrées dans le registre des champions !")
                except Exception as e:
                    print(f"Erreur lors de la sauvegarde du registre : {e}")
                    
        print("\n" + "="*60)
        print("GÉNÉRATION DES GRAPHIQUES DE VARIANTES (INTRA-MODÈLE)")
        print("="*60)
        try:
            plot_model_variants()
            print("Graphiques générés avec succès (voir dossier metrics/)")
        except Exception as e:
            print(f"Erreur lors de la génération des graphiques : {e}")
    elif cfg.MODE == "grid_search":
        for i, exp in enumerate(cfg.EXPERIMENTS_QUEUE):
            cfg.APPROACH = exp["approach"]
            cfg.MODEL_NAME = exp["model"]

            print(f"\n" + "="*60)
            print(f" ANALYSE : {cfg.MODEL_NAME} ({cfg.APPROACH})")
            print(f"Modèle {i+1} sur {len(cfg.EXPERIMENTS_QUEUE)}")
            print("="*60)

            try:
                run_grid_search()
            except Exception as e:
                print(f" Erreur critique lors du Grid Search de {cfg.MODEL_NAME}: {e}")
                continue 

    elif cfg.MODE == "all":
        print(">> Running full pipeline (Download -> Preprocess -> Train -> Eval)...")
        download_from_kaggle()
        run_preprocessing()
        if cfg.APPROACH == "baseline":
            run_training()
        run_evaluation()

    elif cfg.MODE == "compare":
        print(">> Génération des analyses de résultats...")
        # 2. Compare les meilleurs modèles finaux entre eux
        generate_comparison_barchart()

    else:
        print(f"ERROR: The mode '{cfg.MODE}' defined in config.py is unknown.")
        print("Please use 'download', 'preprocess', 'train', 'eval', or 'all'.")

if __name__ == "__main__":
    main()