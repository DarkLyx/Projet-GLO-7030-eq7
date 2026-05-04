import yaml
import os

def load_config(config_path="config/config.yaml"):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Le fichier de configuration {config_path} est introuvable.")
        
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
        
    return config