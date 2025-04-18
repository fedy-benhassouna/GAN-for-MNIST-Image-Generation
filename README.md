# GAN for MNIST Image Generation

This project implements a Generative Adversarial Network (GAN) designed to generate realistic-looking handwritten digit images based on the MNIST dataset. The GAN is composed of a generator and a discriminator trained in opposition: the generator creates images, and the discriminator attempts to distinguish between real and generated images.

## Project Structure

```
.
├── data/
│   └── mnist_data.py      # MNIST data management module
├── models/
│   └── gan.py             # GAN implementation
├── config.py              # Project configuration
├── train.py               # Main training script
├── utils.py               # Utility functions
├── requirements.txt       # Project dependencies
└── README.md              # Documentation
```

## Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

To start training the GAN:
```bash
python train.py
```

The model will automatically save:
- Training checkpoints in the `checkpoints/` directory
- The final trained model in the `saved_models/` directory

## Configuration

The following key parameters can be configured in `config.py`:
- `LATENT_DIM`: Dimension of the noise vector (latent space)
- `LEARNING_RATE`: Learning rate for both generator and discriminator
- `MAX_EPOCHS`: Number of epochs to train the model
- `BATCH_SIZE`: Number of samples per training batch
- `NUM_WORKERS`: Number of subprocesses for data loading

## Results

After training the GAN for 20 epochs, the generator produces synthetic handwritten digits, but they are still easily distinguishable from real ones. This is a typical behavior in early GAN training, as the generator needs more iterations to learn meaningful features.

### Generated Samples After 20 Epochs

The following image showcases 10 generated digits after 20 training epochs:

![Generated digits after 20 epochs](https://github.com/user-attachments/assets/3ea5eeb7-c61a-4459-b6cc-f97b3875c3f0)


Each image was evaluated by the discriminator, which assigned a probability to how likely each was fake. All generated digits were classified as fake with high confidence:

```
Generated image 1 is classified as FAKE with probability: 0.9759586118161678
Generated image 2 is classified as FAKE with probability: 0.9993320878226745
Generated image 3 is classified as FAKE with probability: 0.9930921131744981
Generated image 4 is classified as FAKE with probability: 0.7412363290786743
Generated image 5 is classified as FAKE with probability: 0.8747488409280777
Generated image 6 is classified as FAKE with probability: 0.9975934464018792
Generated image 7 is classified as FAKE with probability: 0.9207934595942497
Generated image 8 is classified as FAKE with probability: 0.9784812442958355
Generated image 9 is classified as FAKE with probability: 0.833181843161583
Generated image 10 is classified as FAKE with probability: 0.9697964619845152
```

> **Interpretation**: With only 20 epochs of training, the GAN is still in the early learning phase. The generator outputs noisy and unrealistic digits, which the discriminator accurately classifies as fake. This highlights the need for longer training durations (typically 100+ epochs) for the generator to capture the data distribution effectively and produce high-quality samples.

