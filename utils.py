import matplotlib.pyplot as plt
import torch
import os
from config import MODEL_SAVE_DIR

def plot_generated_images(model, num_images=9):
    """Génère et affiche des images à partir du générateur"""
    z = torch.randn(num_images, model.hparams.latent_dim).type_as(model.generator.lin1.weight)
    generated_images = model(z).cpu().detach().numpy()

    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    for i, ax in enumerate(axes.flatten()):
        image = generated_images[i].reshape(28, 28)
        ax.imshow(image, cmap='gray')
        ax.axis('off')
    plt.tight_layout()
    plt.show()

def save_model(model, epoch):
    """Sauvegarde le modèle"""
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'generator_state_dict': model.generator.state_dict(),
        'discriminator_state_dict': model.discriminator.state_dict(),
    }, os.path.join(MODEL_SAVE_DIR, f'gan_epoch_{epoch}.pt'))

def load_model(model, path):
    """Charge un modèle sauvegardé"""
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.generator.load_state_dict(checkpoint['generator_state_dict'])
    model.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    return model 