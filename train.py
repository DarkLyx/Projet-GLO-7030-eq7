import os
import torch
import torch.nn as nn
from tqdm import tqdm

from utils.data_loader import get_dataloaders
from utils.visualizer import plot_training_history
from utils.complexity import compute_model_complexity 
from DL_approaches.baselines import build_model, get_optimizer
import config as cfg

def run_training():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Lancement de l'entraînement sur : {device}")

    train_loader, val_loader, _, class_names = get_dataloaders()
    
    model = build_model().to(device)
    optimizer = get_optimizer(model)
    criterion = nn.CrossEntropyLoss()

    input_shape = (1, 3, cfg.IMG_SIZE[0], cfg.IMG_SIZE[1])
    params_fmt, flops_fmt = compute_model_complexity(model, input_shape, device)
    
    print("-" * 50)
    print(f"MÉTRIQUES D'EFFICIENCE DU MODÈLE ({cfg.MODEL_NAME.upper()})")
    print(f"   - Paramètres (Empreinte mémoire) : {params_fmt}")
    print(f"   - FLOPs (Complexité de calcul)   : {flops_fmt}")
    print("-" * 50)

    epochs = cfg.EPOCHS
    save_dir = cfg.CHECKPOINT_DIR
    os.makedirs(save_dir, exist_ok=True)
    best_model_path = os.path.join(save_dir, f"best_{cfg.MODEL_NAME}.pth")

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_loss = float('inf')

    for epoch in range(epochs):
        print(f"\nÉpoque {epoch+1}/{epochs}")
        print("-" * 20)
        
        model.train()
        running_loss, running_corrects = 0.0, 0

        for inputs, labels in tqdm(train_loader, desc="Entraînement"):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc = (running_corrects.double() / len(train_loader.dataset)).item() * 100

        model.eval()
        val_loss, val_corrects = 0.0, 0

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Validation"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                _, preds = torch.max(outputs, 1)
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)

        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_acc = (val_corrects.double() / len(val_loader.dataset)).item() * 100

        print(f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.2f}%")
        print(f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.2f}%")

        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_acc'].append(epoch_val_acc)

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"Nouveau meilleur modèle sauvegardé ! (Loss: {best_val_loss:.4f})")

    plot_training_history(
        history['train_loss'], history['val_loss'], 
        history['train_acc'], history['val_acc'], 
        save_dir=cfg.LOG_DIR  # [MODIFIÉ]
    )
    print("\nEntraînement terminé !")