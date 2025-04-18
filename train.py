import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch

from config import LATENT_DIM, LEARNING_RATE, MAX_EPOCHS, AVAIL_GPUS, RANDOM_SEED
from data.mnist_data import MNISTDataModule
from models.gan import GAN
from utils import plot_generated_images, save_model

def main():
    # Initialisation
    torch.manual_seed(RANDOM_SEED)
    
    # Création des modules
    dm = MNISTDataModule()
    model = GAN(latent_dim=LATENT_DIM, lr=LEARNING_RATE)
    
    # Configuration du checkpoint
    checkpoint_callback = ModelCheckpoint(
        monitor='loss',
        dirpath='checkpoints',
        filename='gan-{epoch:02d}-{loss:.2f}',
        save_top_k=3,
        mode='min',
    )
    
    # Configuration du trainer
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="gpu" if AVAIL_GPUS > 0 else "cpu",
        devices=AVAIL_GPUS,
        callbacks=[checkpoint_callback],
    )
    
    # Entraînement
    trainer.fit(model, dm)
    
    # Visualisation des résultats
    plot_generated_images(model)
    
    # Sauvegarde du modèle final
    save_model(model, MAX_EPOCHS)

if __name__ == "__main__":
    main() 