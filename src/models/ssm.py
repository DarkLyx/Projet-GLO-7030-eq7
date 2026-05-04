import torch
import torch.nn as nn
import configs.config as cfg
from mamba_ssm import Mamba

class VisionMamba(nn.Module):
    """
    Implémentation d'un modèle basé sur les State Space Models (Mamba).
    Les patchs d'images sont traités comme une séquence temporelle.
    """
    def __init__(self, img_size=224, patch_size=16, in_channels=3, num_classes=5, d_model=256):
        super().__init__()
        self.patch_size = patch_size
        num_patches = (img_size // patch_size) ** 2
        
        self.patch_embed = nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)
        
        # Le bloc Mamba remplace le mécanisme d'Attention
        self.mamba_block = Mamba(
            d_model=d_model, # Dimension de l'état caché
            d_state=16,      # Dimension de l'état SSM
            d_conv=4,        # Convolution locale
            expand=2,        # Facteur d'expansion du bloc
        )
        
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # Création des patchs
        x = self.patch_embed(x)             # (B, d_model, H, W)
        x = x.flatten(2).transpose(1, 2)    # (B, num_patches, d_model)
        
        # Traitement séquentiel via State Space Model
        x = self.mamba_block(x)
        x = self.norm(x)
        
        # On utilise la moyenne de la séquence pour la classification (Global Average Pooling)
        x = x.mean(dim=1)
        x = self.head(x)
        return x

def build_ssm_model():
    model_name = cfg.MODEL_NAME.lower()
    if model_name == 'vision_mamba':
        return VisionMamba(img_size=cfg.IMG_SIZE[0], num_classes=cfg.NUM_CLASSES)
    else:
        raise ValueError(f"Modèle SSM '{model_name}' non reconnu.")