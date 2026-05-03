import torch
import torch.nn.functional as F 
import os
from tqdm import tqdm

from src.data.data_loader import get_dataloaders
from src.models.model_factory import build_model
from src.utils.metrics import ModelEvaluator
import configs.config as cfg

def run_evaluation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Lancement de l'évaluation sur : {device}")

    _, _, test_loader, class_names = get_dataloaders()

    model = build_model().to(device)
    best_model_path = os.path.join(cfg.CHECKPOINT_DIR, f"best_{cfg.MODEL_NAME}.pth")
    
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"Aucun modèle trouvé dans {best_model_path}.")
        
    state_dict = torch.load(best_model_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = [] 
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Test des images"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            
            probs = F.softmax(outputs, dim=1) 
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy()) # [NOUVEAU]

    evaluator = ModelEvaluator(class_names=class_names, save_dir=cfg.LOG_DIR)
    evaluator.evaluate_predictions(all_labels, all_preds, all_probs) 
    
    print(f"Évaluation terminée ! Matrice sauvegardée dans {cfg.LOG_DIR}")