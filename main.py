import config as cfg

from download_dataset import download_from_kaggle
from preprocessing import run_preprocessing
from train import run_training
from evaluate import run_evaluation
from dataset_separation import run_dataset_split

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
            
        elif cfg.APPROACH == "mamba":
            print(">> Running State Space Models (Mamba) approach...")
            
        else:
            print(f"ERROR: The approach '{cfg.APPROACH}' is unknown for training.")
            print("Please use 'baseline', 'kan', or 'mamba'.")

    elif cfg.MODE == "eval":
        print(">> Running evaluation and generating confusion matrix...")
        run_evaluation()

    elif cfg.MODE == "all":
        print(">> Running full pipeline (Download -> Preprocess -> Train -> Eval)...")
        download_from_kaggle()
        run_preprocessing()
        if cfg.APPROACH == "baseline":
            run_training()
        run_evaluation()

    else:
        print(f"ERROR: The mode '{cfg.MODE}' defined in config.py is unknown.")
        print("Please use 'download', 'preprocess', 'train', 'eval', or 'all'.")

if __name__ == "__main__":
    main()