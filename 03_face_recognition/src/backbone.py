import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class FaceResNet(nn.Module):
    def __init__(self, embedding_size=128, pretrained=True):
        super(FaceResNet, self).__init__()

        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet18(weights=weights)
        self.backbone.fc = nn.Identity()
        self.projection = nn.Sequential(
            nn.Linear(512, embedding_size),
            nn.BatchNorm1d(embedding_size),
            nn.PReLU(),
        )

    def forward(self, x):
        features = self.backbone(x)
        unnormalized_embeddings = self.projection(features)
        embedding_norm = torch.norm(unnormalized_embeddings, p=2, dim=1, keepdim=True)
        normalized_embeddings = F.normalize(unnormalized_embeddings, p=2, dim=1)

        return normalized_embeddings, embedding_norm
