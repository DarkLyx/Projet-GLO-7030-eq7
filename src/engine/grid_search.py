import configs.config as cfg
from src.engine.train import run_training

PARAM_ABBREV = {
    "LEARNING_RATE": "LR",
    "BATCH_SIZE":    "BS",
    "OPTIMIZER":     "OPT",
    "WEIGHT_DECAY":  "WD",
    "EPOCHS":        "EP",
    "IMG_SIZE":      "IMG",
    "NUM_CLASSES":   "NC",
}


def _format_value(key, value):
    if key == "IMG_SIZE" and isinstance(value, (tuple, list)):
        return "x".join(str(v) for v in value)
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def build_run_name(base_model_name, params):
    parts = [base_model_name]
    for key, value in params.items():
        abbrev = PARAM_ABBREV.get(key, key)
        parts.append(f"{abbrev}{_format_value(key, value)}")
    return "_".join(parts)


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

        run_name = build_run_name(base_model_name, params)
        
        try:
            run_training(run_name=run_name)
            print(f" Terminé : {run_name}")
        except Exception as e:
            print(f" Erreur sur {run_name} : {e}")
            continue 

    print("\n" + "=" * 60)
    print(f"GRID SEARCH TERMINÉ POUR {base_model_name} !")
    print("=" * 60)