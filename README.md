# Deep Convolutional Generative Adversarial Network (DCGAN) for MNIST Data Augmentation

## Project Overview

This project implements a Deep Convolutional Generative Adversarial Network (DCGAN) designed to generate high-quality synthetic handwritten digit images using the MNIST dataset. The primary goals are to:
- Generate realistic synthetic images of handwritten digits
- Provide a robust framework for data augmentation in machine learning tasks

## Key Features

- Advanced DCGAN architecture with improved stability
- High-quality synthetic image generation
- Comprehensive training and visualization tools
- Easy model saving and loading
- Streamlit-based web application for generating and visualizing synthetic images

## Technical Architecture

### Generator Network
The generator creates synthetic images through a series of transposed convolutions:
- **Input**: Random noise vector
- **Layers**: Dense → Reshape → Transposed Convolutions
- **Activation Functions**: LeakyReLU, Batch Normalization
- **Output**: 28x28 grayscale images

### Discriminator Network
The discriminator distinguishes between real and generated images:
- **Input**: 28x28 grayscale images
- **Layers**: Convolutional layers → Dropout → Dense classifier
- **Activation Functions**: LeakyReLU
- **Output**: Binary classification (real vs. synthetic)

## Theoretical Background

### Generative Adversarial Networks (GANs)
Generative Adversarial Networks (GANs) are a class of machine learning frameworks designed by Ian Goodfellow and his colleagues in 2014. GANs consist of two neural networks, the generator and the discriminator, which are trained simultaneously through adversarial processes. The generator aims to produce realistic data, while the discriminator attempts to distinguish between real and synthetic data.

### Deep Convolutional GAN (DCGAN)
DCGAN is a variant of GAN that leverages deep convolutional neural networks (CNNs) to improve the quality of generated images. The key innovations in DCGAN include:
- **Convolutional Layers**: Use of convolutional layers in both the generator and discriminator to capture spatial hierarchies in images.
- **Batch Normalization**: Application of batch normalization to stabilize training and improve convergence.
- **LeakyReLU Activation**: Use of LeakyReLU activation functions to allow gradients to flow through the network, preventing the vanishing gradient problem.
- **Transposed Convolutions**: Use of transposed convolutions (also known as deconvolutions) in the generator to upsample the input noise vector into a full-sized image.

### Training Process
The training process of a GAN involves alternating between training the discriminator and the generator:
1. **Discriminator Training**: The discriminator is trained to maximize the probability of correctly classifying real and synthetic images.
2. **Generator Training**: The generator is trained to minimize the probability of the discriminator correctly classifying synthetic images as fake.

The loss functions used in DCGAN are typically binary cross-entropy losses for both the generator and the discriminator.

## Usage Examples

### Training the Model
```python
from gan import DCGAN
from train_gan import train_gan

# Initialize and train the DCGAN
gan = DCGAN(latent_dim=128)
train_gan(gan, epochs=50)
```

### Generating Synthetic Images
```python
from generate_synthetic_data import generate_synthetic_data
from visualize_syn_data import visualize_synthetic_data

# Generate 1000 synthetic digit images
synthetic_images = generate_synthetic_data(gan.generator)
visualize_synthetic_data(synthetic_images)
```

### Saving and Loading Models
```python
from train_gan import save_gan_model
from load_model import load_gan_model

# Save trained model
save_gan_model(gan, "saved_model_directory")

# Load saved model
loaded_gan = load_gan_model("saved_model_directory")
```

## Streamlit Web Application

The project includes a Streamlit-based web application for generating and visualizing synthetic MNIST digits.

### Running the Application
1. Ensure all dependencies are installed:
    ```sh
    pip install -r requirements.txt
    ```
2. Run the Streamlit application:
    ```sh
    streamlit run app.py
    ```

### Application Features
- **Model Initialization**: Load the pre-trained GAN generator model.
- **Image Generation**: Generate synthetic MNIST digits and visualize them.
- **Image Analysis**: Display statistics about the generated images.

## Potential Applications

1. **Data Augmentation**: Increase training dataset size for digit recognition models.
2. **Anomaly Detection**: Generate diverse synthetic data to improve model robustness.
3. **Machine Learning Research**: Study generative model behavior.
4. **Educational Tool**: Demonstrate generative adversarial network principles.

## Dependencies

- TensorFlow
- NumPy
- Matplotlib
- Streamlit
- PIL (Python Imaging Library)
