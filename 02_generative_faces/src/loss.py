import torch
import torch.nn.functional as F


class VAELoss:
    def __init__(self, kld_weight=0.00025):
        """
        Computes the VAE Loss (Evidence Lower Bound).

        Args:
            kld_weight (float): Weighting factor for the KL divergence term.
                                Since pixel loss is sum over 3*64*64 pixels,
                                KLD usually needs a small weight to balance optimization.
        """
        self.kld_weight = kld_weight

    def __call__(self, recon_x, x, mu, logvar):
        """
        Args:
            recon_x: Reconstructed images (B, C, H, W)
            x: Target images (B, C, H, W)
            mu: Latent mean
            logvar: Latent log variance
        """
        recon_loss = F.binary_cross_entropy(recon_x, x, reduction="sum")
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        total_loss = recon_loss + (self.kld_weight * kld_loss)

        return total_loss, recon_loss, kld_loss
