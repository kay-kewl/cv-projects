# Generative Faces with Convolutional VAE

**Task:** Unsupervised Generative Modeling of human faces using the CelebA dataset.

This project implements a Convolutional Variational Autoencoder. Unlike standard Autoencoders that learn a fixed compression, VAEs learn a probability distribution over the latent space, enabling the generation of new, unseen faces and smooth interpolation between identities.

## Architecture
The model uses a symmetric Encoder-Decoder architecture designed for 64x64 images:
*   **Encoder:** 4-layer Convolutional network with Strided Convolutions. It maps the input image to a Multivariate Gaussian distribution.
*   **Latent Space:** A low-dimensional vector space (default $z=128$) sampled using the Reparameterization Trick.
*   **Decoder:** 4-layer Transposed Convolutional network that reconstructs the image from the latent vector.

## Technical Details
*   **Loss Function:** Evidence Lower Bound, consisting of:
    1.  **Reconstruction Loss:** Binary Cross Entropy to ensure visual fidelity.
    2.  **KL Divergence:** Regularizes the latent space to approximate a Standard Normal Distribution $\mathcal{N}(0, I)$.
*   **Preprocessing:** Center cropping to 178x178 and resizing to 64x64.

## Key Results
*   **Reconstruction:** High fidelity reconstruction of validation images.
*   **Generation:** Generating novel faces from random noise.
*   **Latent Walk:** Smooth morphing between two distinct faces.
