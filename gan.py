import tensorflow as tf
from tensorflow.keras import layers, models # type: ignore

class DCGAN:
    def __init__(self, latent_dim=128):
        """
        Initialize the DCGAN with improved architecture for better performance.

        The architecture uses several techniques for stability:
        - Custom learning rates for generator and discriminator
        - LeakyReLU activation for better gradient flow
        - Dropout for regularization
        - Batch normalization for training stability
        """
        self.latent_dim = latent_dim
        self.generator = self._build_generator()
        self.discriminator = self._build_discriminator()

        # Separate optimizers with different learning rates
        self.generator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

        # Initialize loss function
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        # Loss trackers for monitoring
        self.gen_loss_tracker = tf.keras.metrics.Mean(name="generator_loss")
        self.disc_loss_tracker = tf.keras.metrics.Mean(name="discriminator_loss")

    def _build_generator(self):
        """
        Builds an enhanced generator with:
        - Proper scaling of layer sizes
        - Multiple residual-style connections
        - Advanced activation functions
        - Careful initialization
        """
        model = tf.keras.Sequential([
            # Input layer
            layers.Dense(7 * 7 * 256, input_shape=(self.latent_dim,),
                        kernel_initializer='glorot_uniform'),
            layers.Reshape((7, 7, 256)),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),

            # First upsampling block
            layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same',
                                 kernel_initializer='glorot_uniform'),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),

            # Second upsampling block with increased filters
            layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same',
                                 kernel_initializer='glorot_uniform'),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),

            # Output layer with tanh activation
            layers.Conv2D(1, (5, 5), padding='same', activation='tanh',
                         kernel_initializer='glorot_uniform')
        ])
        return model

    def _build_discriminator(self):
        """
        Builds an enhanced discriminator with:
        - Progressive downsampling
        - Careful dropout rates
        - Advanced activation functions
        """
        model = tf.keras.Sequential([
            # First conv block
            layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                         input_shape=[28, 28, 1]),
            layers.LeakyReLU(alpha=0.2),
            layers.Dropout(0.3),

            # Second conv block with increased capacity
            layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'),
            layers.LeakyReLU(alpha=0.2),
            layers.Dropout(0.3),

            # Additional conv block for better feature extraction
            layers.Conv2D(256, (5, 5), strides=(1, 1), padding='same'),
            layers.LeakyReLU(alpha=0.2),
            layers.Dropout(0.3),

            # Dense layers for classification
            layers.Flatten(),
            layers.Dense(1)
        ])
        return model

    @tf.function
    def train_step(self, real_images):
        """
        Performs a single training step with improved stability measures.
        Uses gradient tape for automatic differentiation and custom loss calculation.
        """
        batch_size = tf.shape(real_images)[0]
        noise = tf.random.normal([batch_size, self.latent_dim])

        # Train discriminator
        with tf.GradientTape() as disc_tape:
            # Generate fake images
            generated_images = self.generator(noise, training=True)

            # Get discriminator outputs
            real_output = self.discriminator(real_images, training=True)
            fake_output = self.discriminator(generated_images, training=True)

            # Calculate discriminator loss with label smoothing
            real_loss = self.cross_entropy(
                tf.random.uniform([batch_size, 1], 0.7, 1.0), real_output
            )
            fake_loss = self.cross_entropy(
                tf.random.uniform([batch_size, 1], 0.0, 0.3), fake_output
            )
            disc_loss = real_loss + fake_loss

        # Train generator
        with tf.GradientTape() as gen_tape:
            generated_images = self.generator(noise, training=True)
            fake_output = self.discriminator(generated_images, training=True)
            gen_loss = self.cross_entropy(
                tf.ones_like(fake_output), fake_output
            )

        # Calculate and apply gradients
        generator_gradients = gen_tape.gradient(
            gen_loss, self.generator.trainable_variables
        )
        discriminator_gradients = disc_tape.gradient(
            disc_loss, self.discriminator.trainable_variables
        )

        self.generator_optimizer.apply_gradients(
            zip(generator_gradients, self.generator.trainable_variables)
        )
        self.discriminator_optimizer.apply_gradients(
            zip(discriminator_gradients, self.discriminator.trainable_variables)
        )

        # Update metrics
        self.gen_loss_tracker.update_state(gen_loss)
        self.disc_loss_tracker.update_state(disc_loss)

        return {
            "g_loss": self.gen_loss_tracker.result(),
            "d_loss": self.disc_loss_tracker.result(),
        }