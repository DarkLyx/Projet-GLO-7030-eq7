import torch
import torch.nn as nn
import torchvision.models as models
import configs.config as cfg

def get_vit_baseline(num_classes=5, pretrained=True):
    """
    Initialise un Vision Transformer (ViT-B/16).
    """
    if pretrained:
        # Utilisation des poids pré-entraînés sur ImageNet (conseillé)
        weights = models.ViT_B_16_Weights.DEFAULT
        model = models.vit_b_16(weights=weights)
    else:
        model = models.vit_b_16(weights=None)
        
    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, num_classes)
    
    return model

def get_swin_transformer(num_classes=5, pretrained=True):
    """
    Initialise un Swin Transformer (Shifted Window) (Swin-T).
    """
    weights = models.Swin_T_Weights.DEFAULT if pretrained else None
    # Swin-T (Tiny) est recommandé pour avoir des FLOPs/Paramètres comparables au ResNet18
    model = models.swin_t(weights=weights)
    in_features = model.head.in_features
    model.head = nn.Linear(in_features, num_classes)
    return model

def build_transformer_model():
    """
    Fonction constructeur pour rester cohérent avec l'archi.
    """
    num_classes = cfg.NUM_CLASSES
    model_name = cfg.MODEL_NAME.lower()
    
    if model_name == 'vit_b_16':
        # On utilise pretrained=True par défaut, car les ViT performent très mal 
        # sans pré-entraînement sur des petits datasets
        return get_vit_baseline(num_classes=num_classes, pretrained=True)
    elif model_name == 'swin_t':
        return get_swin_transformer(num_classes=num_classes, pretrained=True)
    else:
        raise ValueError(f"Modèle Transformer '{model_name}' non reconnu. Essaie 'vit_b_16'.")