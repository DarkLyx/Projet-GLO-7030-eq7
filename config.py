# Use 'download', 'preprocess', 'train', 'eval', or 'all'
MODE = "train" 

# Valid APPROACH: MODEL_NAME...
# baseline: custom_cnn, resnet18
# hybrids: maxvit_t
# transformers: vit_b_16, swin_t
# kan: vision_kan
# ssm: vision_mamba
APPROACH = "transformers"
MODEL_NAME = "swin_t" 

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

LOG_DIR = "rapport/plots/"
CHECKPOINT_DIR = "results/checkpoints/"