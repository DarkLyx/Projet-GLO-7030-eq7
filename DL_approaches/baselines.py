import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import config as cfg # [MODIFIÉ]

class CustomCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(CustomCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

def get_resnet_baseline(num_classes=5):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

def build_baselines_model():
    num_classes = cfg.NUM_CLASSES         
    model_name = cfg.MODEL_NAME         
    
    if model_name == 'custom_cnn':
        return CustomCNN(num_classes=num_classes)
    elif model_name == 'resnet18':
        return get_resnet_baseline(num_classes=num_classes)
    else:
        raise ValueError(f"Modèle {model_name} non reconnu.")
