import torch.nn as nn
import torchvision.models as models
import configs.config as cfg


def get_maxvit_hybrid(num_classes=5, pretrained=True):
    weights = models.MaxVit_T_Weights.DEFAULT if pretrained else None
    # MaxViT-T (Tiny)
    model = models.maxvit_t(weights=weights)
    # La tête du MaxViT est un bloc séquentiel, la dernière couche linéaire est à l'index 5
    in_features = model.classifier[5].in_features
    model.classifier[5] = nn.Linear(in_features, num_classes)
    return model


def build_hybrid_model():
    """
    Aiguille vers le bon modèle Hybride selon la configuration.
    """
    num_classes = cfg.NUM_CLASSES
    model_name = cfg.MODEL_NAME.lower()
    
    if model_name == 'maxvit_t':
        return get_maxvit_hybrid(num_classes=num_classes, pretrained=True)
        
    else:
        raise ValueError(f"Modèle Hybride '{model_name}' non reconnu.")