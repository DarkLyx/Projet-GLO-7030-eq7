MODE = "grid_search" # "download", "preprocess", "train", "eval", "grid_search", "all", "compare"

APPROACH = "baseline" # "baseline", "transformers", "hybrids", "kan", "ssm"
# Modèles disponibles selon l'approche :
# - baseline     : "custom_cnn", "resnet18"
# - transformers : "vit_b_16", "swin_t"
# - hybrids      : "maxvit_t"
# - kan          : "vision_kan"
# - ssm          : "vision_mamba"
MODEL_NAME = "custom_cnn"  

EXPERIMENTS_QUEUE = [
    {"approach": "hybrids", "model": "maxvit_t"},
    {"approach": "baseline", "model": "resnet18"}
]


TASK = "all" 

TASK_CLASSES = {
    "colon": ["colon_aca", "colon_n"],
    "lung":  ["lung_aca", "lung_n", "lung_scc"],
    "all":   ["colon_aca", "colon_n", "lung_aca", "lung_n", "lung_scc"]
}

# --- OPTIMISATION & RÉGULARISATION ---

# 1. Early Stopping
USE_EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 7  # On s'arrête si pas d'amélioration pendant 7 époques

# 2. Learning Rate Scheduler (ReduceLROnPlateau)
USE_LR_SCHEDULER = True
SCHEDULER_FACTOR = 0.5       # On divise le LR par 2
SCHEDULER_PATIENCE = 3       # Après 3 époques de stagnation de la loss

# 3. Gradient Clipping
USE_GRAD_CLIPPING = True
GRAD_CLIP_MAX_NORM = 1.0     # Valeur max pour "couper" les gradients qui explosent

NUM_CLASSES = len(TASK_CLASSES[TASK])

GRID_SEARCH_PARAMS = [
    {"LEARNING_RATE": 0.001,  "BATCH_SIZE": 32},
    {"LEARNING_RATE": 0.0001, "BATCH_SIZE": 32},
    {"LEARNING_RATE": 0.001,  "BATCH_SIZE": 64}
]

EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 0.001
IMG_SIZE = (224, 224)
NUM_CLASSES = 5

OPTIMIZER = "adam" 
WEIGHT_DECAY = 0.0001

RAW_PATH = "data/"
SPLIT_PATH = "data_split/"
PROCESSED_PATH = "data_preprocessed/"

LOG_DIR = "results/logs/"
CHECKPOINT_DIR = "results/checkpoints/"