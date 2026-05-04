import torch
import torch.nn.functional as F 
import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score 

from src.data_processing.data_loader import get_dataloaders
from src.models.model_factory import build_model
from src.utils.metrics import ModelEvaluator
import configs.config as cfg

def run_evaluation(run_name=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    current_run_name = run_name if run_name else cfg.MODEL_NAME
    
    _, _, test_loader, class_names = get_dataloaders()

    model = build_model().to(device)
    best_model_path = os.path.join(cfg.CHECKPOINT_DIR, f"best_{current_run_name}.pth")
    
    if not os.path.exists(best_model_path):
        print(f" Aucun modèle trouvé dans {best_model_path}.")
        return None
        
    state_dict = torch.load(best_model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = [] 
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc=f"Test {current_run_name}"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            
            probs = F.softmax(outputs, dim=1) 
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    evaluator = ModelEvaluator(class_names=class_names, model_name=current_run_name)
    evaluator.evaluate_predictions(all_labels, all_preds, all_probs) 
    
    test_acc = accuracy_score(all_labels, all_preds)
    return test_acc