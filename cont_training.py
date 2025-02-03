from train_gan import save_gan_model
from load_model import load_gan_model

import os
import tensorflow as tf
import numpy as np

def continue_training(gan, additional_epochs=50, batch_size=128, checkpoint_frequency=5):
    """
    Continues training a previously trained GAN model with careful monitoring
    and checkpointing to prevent loss of progress.

    Args:
        gan: Previously trained GAN model
        additional_epochs: Number of additional epochs to train
        batch_size: Batch size for training
        checkpoint_frequency: How often to save checkpoints (in epochs)
    """
    (train_images, _), (_, _) = tf.keras.datasets.mnist.load_data()

    # Reshape and normalize images
    train_images = train_images.reshape(
        train_images.shape[0], 28, 28, 1
    ).astype('float32')
    train_images = (train_images - 127.5) / 127.5

    # Prepare the dataset again for continued training
    dataset = tf.data.Dataset.from_tensor_slices(train_images)\
        .shuffle(60000)\
        .batch(batch_size)\
        .prefetch(tf.data.experimental.AUTOTUNE)

    # Create a directory for checkpoints during continued training
    checkpoint_dir = "gan_continued_training_checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Initialize lists to track losses for monitoring
    generator_losses = []
    discriminator_losses = []

    # Training loop with monitoring and checkpointing
    for epoch in range(additional_epochs):
        print(f"\nContinued Training - Epoch {epoch + 1}/{additional_epochs}")
        epoch_gen_losses = []
        epoch_disc_losses = []

        for batch in dataset:
            # Train step
            metrics = gan.train_step(batch)
            epoch_gen_losses.append(float(metrics['g_loss']))
            epoch_disc_losses.append(float(metrics['d_loss']))

            # Print progress
            print(
                f"G_loss: {metrics['g_loss']:.4f}, "
                f"D_loss: {metrics['d_loss']:.4f}",
                end='\r'
            )

        # Calculate and store average losses for the epoch
        avg_gen_loss = np.mean(epoch_gen_losses)
        avg_disc_loss = np.mean(epoch_disc_losses)
        generator_losses.append(avg_gen_loss)
        discriminator_losses.append(avg_disc_loss)

        # Print epoch summary
        print(f"\nEpoch {epoch + 1} - "
              f"Average G_loss: {avg_gen_loss:.4f}, "
              f"Average D_loss: {avg_disc_loss:.4f}")

        # Save checkpoint if needed
        if (epoch + 1) % checkpoint_frequency == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch + 1}")
            save_gan_model(gan, checkpoint_path)
            print(f"Saved checkpoint at epoch {epoch + 1}")

    return gan, {'generator_losses': generator_losses,
                'discriminator_losses': discriminator_losses}

loaded_gan = load_gan_model("/models/")
# Continue training for more epochs
additional_epochs = 50
new_gan, training_history = continue_training(loaded_gan, additional_epochs)

save_gan_model(new_gan, "/content/")