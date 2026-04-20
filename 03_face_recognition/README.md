# Face Recognition & Quality Estimation with ArcFace

**Task:** Identity Verification and Automated Data Cleaning on the CelebA dataset.

This project implements a robust face recognition system using ArcFace. Unlike standard classification, this model optimizes the embedding space such that faces of the same identity are clustered tightly together on a hypersphere, while different identities are pushed apart.

Additionally, this project introduces a Trash Detection mechanism. By analyzing the magnitude, norm, of unnormalized embeddings, the model can automatically flag low-quality, blurry, or occluded images without explicit supervision.

## Architecture
*   **Backbone:** Pre-trained ResNet18.
*   **Embedding Head:** A linear projection layer mapping features to a 128-dimensional latent space.
*   **Metric Head:** ArcFace mechanism that applies an angular margin penalty ($m=0.5$) during training to enforce intra-class compactness.

## Technical Details
*   **Loss Function:** ArcFace Loss + CrossEntropy.
    $$ L = -\log \frac{e^{s(\cos(\theta_{y_i} + m))}}{e^{s(\cos(\theta_{y_i} + m))} + \sum_{j \neq y_i} e^{s \cos \theta_j}} $$
*   **Inference:**
    1.  **Verification:** Cosine Similarity between two face embeddings.
    2.  **Quality Check:** L2-Norm of the pre-normalized vector serves as a proxy for image quality.

## Key Results
*   **Clear Decision Boundary:** A distinct threshold exists to separate valid matches from impostors.
*   **Automated Filtering:** Successfully identifies "Trash" photos (profiles, occlusions) by filtering the lowest 5% of embedding norms.
