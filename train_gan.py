from gan import DCGAN
from dataset import prepare_dataset

import os
import json

def train_gan(epochs=50, batch_size=128, latent_dim=128):
    """
    Trains the GAN with the specified parameters and returns the trained model.
    """
    # Initialize GAN and prepare dataset
    gan = DCGAN(latent_dim)
    dataset = prepare_dataset(batch_size)

    # Training loop
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        for batch in dataset:
            metrics = gan.train_step(batch)
            print(
                f"G_loss: {metrics['g_loss']:.4f}, "
                f"D_loss: {metrics['d_loss']:.4f}",
                end='\r'
            )

    return gan

def save_gan_model(gan, save_dir="saved_gan_model"):
    """
    Saves all components of the GAN model.

    Args:
        gan: The trained GAN model instance
        save_dir: Directory where the model will be saved
    """
    # Create the save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Save the generator
    generator_path = os.path.join(save_dir, "generator.keras")
    gan.generator.save(generator_path)

    # Save the discriminator
    discriminator_path = os.path.join(save_dir, "discriminator.keras")
    gan.discriminator.save(discriminator_path)

    # Save the configuration
    config = {
        "latent_dim": gan.latent_dim,
        "generator_optimizer": {
            "learning_rate": float(gan.generator_optimizer.learning_rate.numpy()),  # Convert to float64
            "beta_1": gan.generator_optimizer.beta_1
        },
        "discriminator_optimizer": {
            "learning_rate": float(gan.discriminator_optimizer.learning_rate.numpy()), # Convert to float64
            "beta_1": gan.discriminator_optimizer.beta_1
        }
    }

    config_path = os.path.join(save_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

# Training configuration
EPOCHS = 2
BATCH_SIZE = 128
LATENT_DIM = 128

# Train the model
gan = train_gan(EPOCHS, BATCH_SIZE, LATENT_DIM)

# gan.generator.save('generator_model.keras')
# gan.discriminator.save('discriminator_model.keras')
save_gan_model(gan, "/models/")
