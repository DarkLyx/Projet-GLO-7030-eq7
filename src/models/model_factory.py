import torch.optim as optim
import configs.config as cfg
from src.models.baselines import build_baselines_model
from src.models.transformers import build_transformer_model
from src.models.hybrids import build_hybrid_model
from src.models.kan import build_kan_model
from src.models.ssm import build_ssm_model

def build_model():

    approach = cfg.APPROACH.lower()
    
    if approach == "baseline":
        return build_baselines_model()
        
    elif approach == "transformers":
        return build_transformer_model()
    
    elif approach == "hybrids":
        return build_hybrid_model()
        
    elif approach == "kan":
        return build_kan_model()
        
    elif approach == "ssm":
        return build_ssm_model()
        
    else:
        raise ValueError(f"Approche non reconnue : {approach}")
    

def get_optimizer(model):
    """
    Génère l'optimiseur configuré pour le modèle passé en paramètre.
    """
    opt_name = cfg.OPTIMIZER.lower()      
    lr = cfg.LEARNING_RATE               
    weight_decay = cfg.WEIGHT_DECAY      

    if opt_name == 'adam':
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_name == 'adamw':
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_name == 'sgd':
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError(f"Optimiseur {opt_name} non supporté.")