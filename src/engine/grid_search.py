import configs.config as cfg
from src.engine.train import run_training

def run_grid_search():
    print("=" * 60)
    print(f"LANCEMENT DU GRID SEARCH POUR : {cfg.MODEL_NAME}")
    print(f"Nombre de combinaisons à tester : {len(cfg.GRID_SEARCH_PARAMS)}")
    print("=" * 60)

    base_model_name = cfg.MODEL_NAME

    for i, params in enumerate(cfg.GRID_SEARCH_PARAMS):
        print(f"\n\n>>> EXPÉRIENCE [{i+1}/{len(cfg.GRID_SEARCH_PARAMS)}]")
        print(f">>> Paramètres appliqués : {params}")

        # 1. Mise à jour dynamique des variables dans config.py
        for key, value in params.items():
            setattr(cfg, key, value)

        # 2. Création du nom unique pour cette expérience
        run_name = f"{base_model_name}_LR{cfg.LEARNING_RATE}_BS{cfg.BATCH_SIZE}"
        
        # 3. Lancement de l'entraînement
        run_training(run_name=run_name)

    print("\n" + "=" * 60)
    print("GRID SEARCH TERMINÉ !")
    print("Consultez le fichier global_complexity_comparison.csv et le dossier metrics/")