# src/config.py

import torch

# Define device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Dataset paths
ROOT_DIR = 'dataset'
IMAGES_DIR_TRAIN = 'dataset/images/train'
IMAGES_DIR_VAL = 'dataset/images/val'
IMAGES_DIR_TEST = 'dataset/images/test'
ANNOTATIONS_TRAIN = 'dataset/annotations/instances_train.json'
ANNOTATIONS_VAL = 'dataset/annotations/instances_val.json'
ANNOTATIONS_TEST = 'dataset/annotations/instances_test.json'

# Model hyperparameters
NUM_CLASSES = 5  # 4 defects + 1 background
LEARNING_RATE = 0.005
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
BATCH_SIZE = 2
NUM_EPOCHS = 10

# Label mapping for defects
LABEL_MAP = {
    "background": 0,
    "scratch": 1,
    "dent": 2,
    "pit": 3,
    "cut": 4
}

# Diameter ranges and conversion factor
# NOTE: You will need to calibrate PIXEL_TO_CM_RATIO for your specific images
PIXEL_TO_CM_RATIO = 0.01  # Example: 100 pixels = 1 cm
INNER_DIAMETER_RANGE = (0.5, 1.5)  # in cm
OUTER_DIAMETER_RANGE = (2.5, 3.5)  # in cm