# Semantic Segmentation with U-Net and ResNet50 Encoder

**Task:** Binary Semantic Segmentation of birds from the CUB-200 dataset.

This project implements a U-Net architecture from scratch, utilizing a pre-trained ResNet50 as the feature encoder (Transfer Learning). The goal is to produce pixel-perfect binary masks separating the bird from the background.

## Architecture
The model uses a classic Encoder-Decoder structure:
*   **Encoder:** ResNet50 (pre-trained on ImageNet). We extract features from layers `conv1`, `layer1`, `layer2`, `layer3`, and `layer4`.
*   **Decoder:** Custom upsampling blocks with skip connections to recover spatial resolution.
*   **Head:** A final 1x1 convolution outputs the binary logits.

## Pipeline
*   **Augmentation:** Heavy use of `Albumentations` (Elastic Transform, Coarse Dropout, Color Jitter) to prevent overfitting on the relatively small dataset.
*   **Loss Function:** `BCEWithLogitsLoss` for numerical stability.
*   **Metric:** Intersection over Union (IoU).
