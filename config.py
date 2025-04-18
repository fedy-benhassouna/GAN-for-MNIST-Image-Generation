import os

# Configuration générale
RANDOM_SEED = 42
BATCH_SIZE = 128
NUM_WORKERS = int(os.cpu_count() / 2)
AVAIL_GPUS = 1

# Configuration du GAN
LATENT_DIM = 100
LEARNING_RATE = 0.0002
MAX_EPOCHS = 50

# Chemins des données
DATA_DIR = "./data"
MODEL_SAVE_DIR = "./saved_models" 