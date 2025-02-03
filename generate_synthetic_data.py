import numpy as np
import os
import tensorflow as tf

from load_model import load_gan_model

# Function to generate and visualize images
def generate_synthetic_data(gan, num_samples=1000, output_dir="synthetic_data"):
    """
    Generates synthetic MNIST images and saves them along with their predicted labels.

    Args:
        gan: The loaded GAN model
        num_samples: Number of synthetic images to generate
        output_dir: Directory to save the synthetic data
    """
    os.makedirs(output_dir, exist_ok=True)

    # Generate synthetic images
    noise = tf.random.normal([num_samples, gan.latent_dim])
    generated_images = gan.generator(noise, training=False)

    # Convert to numpy and rescale to [0, 255]
    generated_images = ((generated_images + 1) * 127.5).numpy().astype('uint8')

    # Save the synthetic data
    np.save(os.path.join(output_dir, "synthetic_images.npy"), generated_images)
    print(f"Saved {num_samples} synthetic images to {output_dir}")
    return generated_images

loaded_gan = load_gan_model("/models/")
synthetic_data = generate_synthetic_data(loaded_gan, num_samples=1000, output_dir="synthetic_data")