import tensorflow as tf

def prepare_dataset(batch_size=128):
    """
    Prepares the MNIST dataset with proper preprocessing and optimization.
    """
    # Load and preprocess MNIST data
    (train_images, _), (_, _) = tf.keras.datasets.mnist.load_data()

    # Reshape and normalize images
    train_images = train_images.reshape(
        train_images.shape[0], 28, 28, 1
    ).astype('float32')
    train_images = (train_images - 127.5) / 127.5

    # Create efficient dataset pipeline
    return tf.data.Dataset.from_tensor_slices(train_images)\
        .shuffle(60000)\
        .batch(batch_size)\
        .prefetch(tf.data.experimental.AUTOTUNE)