MODE = "train" 

APPROACH = "hybrids"
MODEL_NAME = "maxvit_t" 

TASK = "all" 

TASK_CLASSES = {
    "colon": ["colon_aca", "colon_n"],
    "lung":  ["lung_aca", "lung_n", "lung_scc"],
    "all":   ["colon_aca", "colon_n", "lung_aca", "lung_n", "lung_scc"]
}

NUM_CLASSES = len(TASK_CLASSES[TASK])

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