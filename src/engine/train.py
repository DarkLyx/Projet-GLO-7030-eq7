import os
import torch
import torch.nn as nn
from tqdm import tqdm
import pandas as pd
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.data_processing.data_loader import get_dataloaders
from src.utils.visualizer import plot_training_history
from src.utils.complexity import compute_model_complexity 
from src.models.model_factory import build_model, get_optimizer
import configs.config as cfg

def save_efficiency_metrics(model_name, params, flops, base_save_dir="results/metrics"):
    """Sauvegarde les FLOPs et Paramètres dans un fichier global."""
    comp_dir = os.path.join(base_save_dir, "complexity")
    os.makedirs(comp_dir, exist_ok=True)
    
    global_csv_path = os.path.join(comp_dir, "global_complexity_comparison.csv")
    new_data = pd.DataFrame({'model': [model_name], 'parameters': [params], 'flops': [flops]})
    
    if os.path.exists(global_csv_path):
        df = pd.read_csv(global_csv_path)
        if model_name in df['model'].values:
            df.loc[df['model'] == model_name, ['parameters', 'flops']] = [params, flops]
        else:
            df = pd.concat([df, new_data], ignore_index=True)
    else:
        df = new_data
        
    df.to_csv(global_csv_path, index=False)


def run_training(run_name=None):
    current_run_name = run_name if run_name else cfg.MODEL_NAME

    print(f"PyTorch Version: {torch.__version__}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" Lancement de l'entraînement sur : {device}")
    print(f"Modèle actif : {current_run_name}")

    train_loader, val_loader, _, class_names = get_dataloaders()
    
    model = build_model().to(device)
    optimizer = get_optimizer(model)
    criterion = nn.CrossEntropyLoss()

    scheduler = None
    if getattr(cfg, 'USE_LR_SCHEDULER', False):
        scheduler = ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=cfg.SCHEDULER_FACTOR, 
            patience=cfg.SCHEDULER_PATIENCE
        )
        print(f" LR Scheduler activé (Factor: {cfg.SCHEDULER_FACTOR}, Patience: {cfg.SCHEDULER_PATIENCE})")

    input_shape = (1, 3, cfg.IMG_SIZE[0], cfg.IMG_SIZE[1])
    params_fmt, flops_fmt = compute_model_complexity(model, input_shape, device)
    save_efficiency_metrics(current_run_name, params_fmt, flops_fmt)

    epochs = cfg.EPOCHS
    save_dir = cfg.CHECKPOINT_DIR
    os.makedirs(save_dir, exist_ok=True)
    best_model_path = os.path.join(save_dir, f"best_{current_run_name}.pth")

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_loss = float('inf')
    
    epochs_no_improve = 0
    early_stop = False

    for epoch in range(epochs):
        if early_stop:
            break 

        print(f"\nÉpoque {epoch+1}/{epochs}")
        print("-" * 20)
        
        # --- PHASE D'ENTRAÎNEMENT ---
        model.train()
        running_loss, running_corrects = 0.0, 0

        for inputs, labels in tqdm(train_loader, desc="Entraînement"):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()

            if getattr(cfg, 'USE_GRAD_CLIPPING', False):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.GRAD_CLIP_MAX_NORM)

            optimizer.step()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc = (running_corrects.double() / len(train_loader.dataset)).item() * 100

        # --- PHASE DE VALIDATION ---
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

        if scheduler is not None:
            old_lr = optimizer.param_groups[0]['lr']
            scheduler.step(epoch_val_loss)
            new_lr = optimizer.param_groups[0]['lr']
            if new_lr < old_lr:
                print(f" Le plateau a été atteint. Nouveau Learning Rate : {new_lr}")

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            epochs_no_improve = 0  # On remet le compteur à zéro
            torch.save(model.state_dict(), best_model_path)
            print(f"Nouveau meilleur modèle sauvegardé ! (Loss: {best_val_loss:.4f})")
        else:
            epochs_no_improve += 1
            print(f"Pas d'amélioration de la validation depuis {epochs_no_improve} époque(s).")
            
            if getattr(cfg, 'USE_EARLY_STOPPING', False) and epochs_no_improve >= cfg.EARLY_STOPPING_PATIENCE:
                print(f" EARLY STOPPING déclenché ! Fin de l'entraînement pour {current_run_name}.")
                early_stop = True

    history_dir = os.path.join("results/metrics", "training_history")
    plot_training_history(history=history, save_dir=history_dir, model_name=current_run_name)
    print("\nEntraînement terminé !")