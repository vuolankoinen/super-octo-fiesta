# Paths
DATA_DIR = "data"
TRAIN_DIR = "data/train"
VAL_DIR = "data/validate"
TEST_DIR = "data/test"
MODEL_DIR = "model"
LOG_DIR = "log"

# Data preparation and training
TRAIN_VAL_SPLIT = (0.7, 0.15, 0.15)
NUM_CATEGORIES = 3  # Take the first n from the following
CATEGORIES = [
    "Forest",
    "Field",
    "Other",
][:NUM_CATEGORIES]
IMG_SIZE = (720, 1280)
BATCH_SIZE = 4
