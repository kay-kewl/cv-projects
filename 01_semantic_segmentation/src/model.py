import torch
import torch.nn as nn
import torchvision.models as models


class DecoderBlock(nn.Module):
    """
    Standard U-Net decoder block: Upsample -> Concat -> Conv -> BN -> ReLU
    """

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=False
        )
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels + skip_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = self.upsample(x)
        if x.shape != skip.shape:
            x = torch.nn.functional.interpolate(
                x, size=skip.shape[2:], mode="bilinear", align_corners=False
            )
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UnetResNet50(nn.Module):
    def __init__(self, n_classes=1):
        super().__init__()

        weights = models.ResNet50_Weights.DEFAULT
        backbone = models.resnet50(weights=weights)

        self.enc1 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.pool = backbone.maxpool
        self.enc2 = backbone.layer1
        self.enc3 = backbone.layer2
        self.enc4 = backbone.layer3
        self.enc5 = backbone.layer4

        self.dec1 = DecoderBlock(
            in_channels=2048, skip_channels=1024, out_channels=1024
        )
        self.dec2 = DecoderBlock(in_channels=1024, skip_channels=512, out_channels=512)
        self.dec3 = DecoderBlock(in_channels=512, skip_channels=256, out_channels=256)
        self.dec4 = DecoderBlock(in_channels=256, skip_channels=64, out_channels=128)

        self.final_upsample = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=False
        )
        self.final_conv = nn.Conv2d(128, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.enc1(x)
        x_pool = self.pool(x1)
        x2 = self.enc2(x_pool)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        x5 = self.enc5(x4)

        d1 = self.dec1(x5, x4)
        d2 = self.dec2(d1, x3)
        d3 = self.dec3(d2, x2)
        d4 = self.dec4(d3, x1)

        out = self.final_upsample(d4)
        logits = self.final_conv(out)

        return logits
