# Computer Vision Projects

A collection of Computer Vision projects focusing on deep learning architectures for Segmentation, Generative Modeling, and Metric Learning. This repository demonstrates the implementation of custom U-Net decoders, Variational Autoencoders with KL-Divergence optimization, and robust identity verification systems using ArcFace.

## Projects

### 1. [Semantic Segmentation & Transfer Learning](./01_semantic_segmentation)
*   **Custom U-Net Architecture:** Engineered a U-Net model using a pre-trained ResNet50 encoder and a custom-built decoder to handle complex feature concatenation;
*   **Robust Pipeline:** Implemented an aggressive augmentation pipeline using Albumentations with ElasticTransform and CoarseDropout to prevent overfitting on the CUB-200 dataset;
*   **Results:** Achieved 0.85 IoU on the test set, demonstrating precise object localization.

### 2. [Generative Modeling with VAEs](./02_generative_faces)
**Focus:** Variational Inference, Latent Space Analysis, and Convolutional Networks.

*   **Convolutional VAE:** Implemented a VAE from scratch using Strided Convolutions for Encoder and Transposed Convolutions for Decoder to preserve spatial hierarchies;
*   **Latent Space Arithmetic:** Optimized the Evidence Lower Bound, or ELBO, with a custom KL-Divergence loss, enabling smooth interpolation between face embeddings;
*   **Sampling:** Demonstrated generation of novel photorealistic faces by sampling from the learned multivariate Gaussian distribution.

### 3. [Face Recognition & Metric Learning](./03_face_recognition)
**Focus:** Vector Space Optimization, ArcFace Loss, and Quality Estimation.

*   **Metric Learning:** Trained a ResNet18 backbone using ArcFace Loss to maximize inter-class separability and intra-class compactness on the CelebA dataset;
*   **Quality Estimation:** Developed an algorithm to automatically filter low-quality/corrupted images by analyzing the norm distribution of unnormalized embeddings;
*   **Analysis:** Visualized cosine similarity distributions to validate the decision boundary between "Same" and "Different" identities.

## Setup

1. Clone the repository:
  ```bash
    git clone https://github.com/kay-kewl/cv-projects.git
    cd cv-projects
  ```

2. Create and activate a virtual environment:
  ```bash
    python -m venv venv

    # windows 
    .\venv\Scripts\activate
    # macos/linux
    source venv/bin/activate
  ```
  
3. Install dependencies:
  ```
    pip install -r requirements.txt
  ```
