from gan import DCGAN

import os
import json
import tensorflow as tf

def load_gan_model(load_dir="saved_gan_model"):
    """
    Loads a previously saved GAN model, including:
    - Generator architecture and weights
    - Discriminator architecture and weights
    - Model configuration

    Args:
        load_dir: Directory containing the saved model

    Returns:
        Loaded GAN model instance
    """
    # Load configuration
    config_path = os.path.join(load_dir, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)

    # Create a new GAN instance with saved configuration
    gan = DCGAN(latent_dim=config["latent_dim"])

    # Load saved models
    generator_path = os.path.join(load_dir, "generator.keras")
    discriminator_path = os.path.join(load_dir, "discriminator.keras")

    gan.generator = tf.keras.models.load_model(generator_path)
    gan.discriminator = tf.keras.models.load_model(discriminator_path)

    # Recreate optimizers with saved configurations
    gan.generator_optimizer = tf.keras.optimizers.Adam(
        learning_rate=config["generator_optimizer"]["learning_rate"],
        beta_1=config["generator_optimizer"]["beta_1"]
    )
    gan.discriminator_optimizer = tf.keras.optimizers.Adam(
        learning_rate=config["discriminator_optimizer"]["learning_rate"],
        beta_1=config["discriminator_optimizer"]["beta_1"]
    )

    return gan

# loaded_gan = load_gan_model("/models/")