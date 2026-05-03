import configs.config as cfg

from src.data_processing.download_dataset import download_from_kaggle
from src.data_processing.preprocessing import run_preprocessing
from src.engine.train import run_training
from src.engine.evaluate import run_evaluation
from src.data_processing.dataset_separation import run_dataset_split
from src.engine.grid_search import run_grid_search
from src.utils.compare import generate_comparison_barchart
from src.utils.compare_modeltype import plot_model_variants
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
        print(">> Running evaluation and generating confusion matrix...")
        run_evaluation()

    elif cfg.MODE == "grid_search":
        for i, exp in enumerate(cfg.EXPERIMENTS_QUEUE):
            # 1. On définit l'architecture actuelle pour TOUT le projet
            cfg.APPROACH = exp["approach"]
            cfg.MODEL_NAME = exp["model"]

            print(f"\n" + "="*60)
            print(f" ANALYSE : {cfg.MODEL_NAME} ({cfg.APPROACH})")
            print(f"Modèle {i+1} sur {len(cfg.EXPERIMENTS_QUEUE)}")
            print("="*60)

            # 2. On appelle le moteur de Grid Search. 
            # Note : On ne passe PAS de run_name ici, car grid_search le génère lui-même.
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
        # 1. Compare les variantes d'un même modèle entre elles
        plot_model_variants()
        # 2. Compare les meilleurs modèles finaux entre eux
        generate_comparison_barchart()

    else:
        print(f"ERROR: The mode '{cfg.MODE}' defined in config.py is unknown.")
        print("Please use 'download', 'preprocess', 'train', 'eval', or 'all'.")

if __name__ == "__main__":
    main()