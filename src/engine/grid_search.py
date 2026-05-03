import configs.config as cfg
from src.engine.train import run_training

def run_grid_search():
    """
    Lance une série d'entraînements pour le modèle actuellement défini 
    dans cfg.MODEL_NAME en utilisant la grille d'hyperparamètres.
    """
    print("\n" + "-" * 30)
    print(f"DÉMARRAGE GRID SEARCH : {cfg.MODEL_NAME}")
    print(f"Famille d'approche   : {cfg.APPROACH}")
    print(f"Nombre de combinaisons : {len(cfg.GRID_SEARCH_PARAMS)}")
    print("-" * 30)

    base_model_name = cfg.MODEL_NAME

    for i, params in enumerate(cfg.GRID_SEARCH_PARAMS):
        print(f"\n[Combo {i+1}/{len(cfg.GRID_SEARCH_PARAMS)}] Paramètres : {params}")

        for key, value in params.items():
            setattr(cfg, key, value)

        run_name = f"{base_model_name}_LR{cfg.LEARNING_RATE}_BS{cfg.BATCH_SIZE}"
        
        try:
            run_training(run_name=run_name)
            print(f" Terminé : {run_name}")
        except Exception as e:
            print(f" Erreur sur {run_name} : {e}")
            continue 

    print("\n" + "=" * 60)
    print(f"GRID SEARCH TERMINÉ POUR {base_model_name} !")
    print("=" * 60)