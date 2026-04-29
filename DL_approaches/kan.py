import torch
import torch.nn as nn
import config as cfg

#Luigos pense bien à installer le repo github du ImportError sur ICE

# On importe la couche KAN linéaire (à installer via la communauté)
try:
    from efficient_kan import KAN as KANLinear
except ImportError:
    print("ATTENTION: La librairie 'efficient_kan' n'est pas installée.")
    print("Tapez: pip install git+https://github.com/Blealtan/efficient-kan.git")

class VisionKAN(nn.Module):
    """
    Une implémentation simplifiée d'un Vision KAN.
    On découpe l'image en patchs (comme un ViT), puis on passe les caractéristiques 
    à travers des couches de Kolmogorov-Arnold au lieu de couches d'attention.
    """
    def __init__(self, img_size=224, patch_size=16, in_channels=3, num_classes=5, embed_dim=256):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # Projection des patchs
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.flatten = nn.Flatten(2)
        
        # Réseau KAN (remplace les MLP/Transformers classiques)
        # On utilise une couche KAN pour réduire la dimension, puis classifier
        self.kan_block = KANLinear([embed_dim * self.num_patches, 128, num_classes])

    def forward(self, x):
        # x shape: (Batch, Channels, Height, Width)
        x = self.proj(x)                  # -> (Batch, EmbedDim, H/patch, W/patch)
        x = self.flatten(x)               # -> (Batch, EmbedDim, NumPatches)
        x = x.transpose(1, 2)             # -> (Batch, NumPatches, EmbedDim)
        x = x.reshape(x.shape[0], -1)     # -> (Batch, NumPatches * EmbedDim)
        
        # Classification via les fonctions splines du KAN
        x = self.kan_block(x)
        return x

def build_kan_model():
    model_name = cfg.MODEL_NAME.lower()
    if model_name == 'vision_kan':
        return VisionKAN(img_size=cfg.IMG_SIZE[0], num_classes=cfg.NUM_CLASSES)
    else:
        raise ValueError(f"Modèle KAN '{model_name}' non reconnu.")