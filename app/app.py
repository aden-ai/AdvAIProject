import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import io
from PIL import Image

def generate_synthetic_data(generator, num_samples=1000, latent_dim=128):
    noise = tf.random.normal([num_samples, latent_dim])
    generated_images = generator(noise, training=False)
    generated_images = ((generated_images + 1) * 127.5).numpy().astype('uint8')
    return generated_images

def plot_generated_images(images):
    num_images = len(images)
    cols = min(5, num_images)
    rows = (num_images - 1) // cols + 1
    
    plt.figure(figsize=(15, 3 * rows))
    for i in range(num_images):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(images[i, :, :, 0], cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return buf

def analyze_generated_images(images):
    return {
        "Total Images": len(images),
        "Mean Pixel Value": np.mean(images),
        "Pixel Value Std Dev": np.std(images),
        "Min Pixel Value": np.min(images),
        "Max Pixel Value": np.max(images)
    }

def load_generator_model():
    return tf.keras.models.load_model("app/models/generator.keras", compile=False)

def main():
    st.set_page_config(
        page_title="SynthNet: MNIST GAN Generator",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    hide_streamlit_style = """
        <style>
        #MainMenu {visibility: hidden;} /* Hides the main menu */
        footer {visibility: hidden;} /* Hides the footer */
        header {visibility: hidden;} /* Hides the header */
        .css-1d391kg {visibility: hidden;} /* Hides the status indicator */
        .css-1v3fvcr {visibility: hidden;} /* Hides the Streamlit watermark */
        .css-1v0mbdj {visibility: hidden;} /* Hides the overall container */
        </style>
        """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)
    
    st.markdown("""
    <style>
    .big-font {
        font-size:20px !important;
        font-weight: bold;
        color: #2C3E50;
    }
    .sidebar .sidebar-content {
        background-color: #F0F2F6;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("🤖 SynthNet: MNIST Digit Generator")
    st.markdown("""
    ### Synthetic Digit Generation using Deep Convolutional Generative Adversarial Network
    
    Generate realistic handwritten digits using advanced machine learning techniques.
    """)

    image = Image.open("app/images/banner.png")
    st.image(image, caption="Sample Generated Images")
    
    # Initialize session state for generator
    if 'generator' not in st.session_state:
        st.session_state.generator = None
        
    st.markdown("---")
    st.header("Image Generation")

    st.sidebar.header("🛠️ Model Configuration")

    # Dynamically show load button only if model is not loaded
    if st.session_state.generator is None:
        if st.sidebar.button("🚀 Initialize GAN Generator"):
            try:
                st.session_state.generator = load_generator_model()
                st.rerun()  # Rerun to update the sidebar
            except Exception as e:
                st.sidebar.error(f"🚨 Model Initialization Failed: {e}")
    else:
        #Optional unload button if you want to reset
        if st.sidebar.button("🔌 Unload Model"):
            st.session_state.generator = None
            st.rerun()

    st.sidebar.markdown("<br>", unsafe_allow_html=True)  # Adds two line breaks

    num_images = st.sidebar.slider(
        "Number of Images to Generate", 
        min_value=1, 
        max_value=1000, 
        value=10
    )

    if st.button("Generate Synthetic Digits"):
        if st.session_state.generator is not None:
            try:
                images = generate_synthetic_data(
                    st.session_state.generator, 
                    num_images
                )
                st.subheader("Generated Digits")
                image_buffer = plot_generated_images(images)
                st.image(image_buffer)
                
                st.subheader("Image Analysis")
                stats = analyze_generated_images(images)
                
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    for key, value in list(stats.items())[:1]:
                        st.metric(key, f"{value:.2f}")
                with col2:
                    for key, value in list(stats.items())[1:2]:
                        st.metric(key, f"{value:.2f}")
                with col3:
                    for key, value in list(stats.items())[2:3]:
                        st.metric(key, f"{value:.2f}")
                with col4:
                    for key, value in list(stats.items())[3:4]:
                        st.metric(key, f"{value:.2f}")
                with col5:
                    for key, value in list(stats.items())[4:]:
                        st.metric(key, f"{value:.2f}")
            
            except Exception as e:
                st.error(f"Error Generating Images: {e}")
        else:
            st.warning("Please load the model first!")
    
    st.sidebar.header("📖 About SynthNet")
    st.sidebar.info("""
    SynthNet is a Deep Convolutional GAN for generating synthetic MNIST digits.
    
    Key Features:
    - High-quality digit generation
    - Configurable latent space
    - Advanced machine learning techniques
    """)


# if __name__ == "__main__":
#     main()
main()
