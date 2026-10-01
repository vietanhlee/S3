"""
specimen_invariance_framework/datasets/augmentations.py
======================================================
Data augmentations for macroscopic cross-sectional captures:
- Standard spatial and color jitter transformations.
- Style-randomization via Fourier Amplitude Mixing (FDA/FACT) and AdaIN perturbations.
"""

import math
import random
import torch
import torch.nn as nn
from torchvision import transforms


def build_train_transform(
    image_size: int = 224,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
):
    """Standard training transformations."""
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def build_val_transform(
    image_size: int = 224,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
):
    """Deterministic validation / test transformations."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


class FourierAmplitudeMixing(nn.Module):
    """
    Fourier Amplitude Mixing for Domain/Specimen Style-Randomization.
    
    Perturbs low-level illumination, color temperature, and surface artifacts
    by interpolating the amplitude spectrum with a shuffled partner in the batch,
    while strictly preserving the phase spectrum (which encodes structural xylem anatomy).
    """
    def __init__(self, alpha: float = 0.5, p: float = 0.5):
        super().__init__()
        self.alpha = alpha
        self.p = p

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, C, H, W)
        Returns:
            x_mixed: Tensor of shape (B, C, H, W)
        """
        if not self.training or random.random() > self.p or x.size(0) < 2:
            return x

        B, C, H, W = x.shape
        # Compute 2D real FFT
        fft = torch.fft.rfft2(x, norm="ortho")
        amp = torch.abs(fft)
        phase = torch.angle(fft)

        # Shuffle amplitude across batch
        perm = torch.randperm(B, device=x.device)
        amp_perm = amp[perm]

        # Draw mixing coefficient from Beta distribution
        lam = torch.distributions.Beta(self.alpha, self.alpha).sample((B, 1, 1, 1)).to(x.device)
        lam = torch.clamp(lam, 0.0, 0.5)  # Keep original image dominant

        # Interpolate amplitude spectrum
        amp_mixed = (1.0 - lam) * amp + lam * amp_perm

        # Recombine with original phase spectrum
        fft_mixed = torch.polar(amp_mixed, phase)
        x_mixed = torch.fft.irfft2(fft_mixed, s=(H, W), norm="ortho")

        return x_mixed
