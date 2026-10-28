"""Deterministic stain / scanner acquisition profiles.

In real computational pathology, the dominant cross-site shift is
acquisition: different scanners have different white balance and
color response, and different labs stain H&E slides with different
intensity. PCam ships no per-patch scanner metadata, so this module
SIMULATES acquisition shift with one fixed, deterministic color
profile per simulated hospital:

    out = clip( contrast * (gains * rgb - luma_mean) ... )

Concretely, per profile (channel gains g, saturation s, brightness
b, contrast c), applied to a float image in [0, 1]:

    1. white balance:  rgb <- rgb * g            (per-channel gain)
    2. saturation:     rgb <- luma + s * (rgb - luma)
    3. contrast:       rgb <- (rgb - 0.5) * c + 0.5
    4. brightness:     rgb <- rgb + b
    5. clip to [0, 1]

Client 0 is the identity (reference scanner / reference lab).
Profiles are constants, not random draws: the same patch under the
same profile always yields the same pixels (verified by tests).
These are acquisition SIMULATIONS, not measurements of any real
scanner or lab.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StainProfile:
    name: str
    gains: tuple[float, float, float]  # per-channel white-balance gains
    saturation: float
    brightness: float
    contrast: float


PROFILES: tuple[StainProfile, ...] = (
    StainProfile("reference", (1.00, 1.00, 1.00), 1.00, 0.00, 1.00),
    StainProfile("warm_scanner", (1.06, 0.97, 0.90), 1.15, 0.02, 1.05),
    StainProfile("cool_scanner", (0.94, 1.00, 1.08), 0.85, -0.02, 0.95),
    StainProfile("overstained_lab", (1.03, 1.03, 0.94), 1.25, 0.04, 1.10),
)

# ITU-R BT.601 luma weights.
_LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)


def apply_profile(images: np.ndarray, profile: StainProfile) -> np.ndarray:
    """Apply a stain profile to uint8 images of shape (N, H, W, 3).

    Returns uint8 images of the same shape. Pure function of the
    inputs: no RNG, no global state.
    """
    if images.dtype != np.uint8:
        raise TypeError(f"expected uint8 images, got {images.dtype}")
    if images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError(f"expected (N, H, W, 3), got {images.shape}")

    rgb = images.astype(np.float32) / 255.0
    gains = np.array(profile.gains, dtype=np.float32)
    rgb = rgb * gains
    luma = (rgb * _LUMA).sum(axis=-1, keepdims=True)
    rgb = luma + profile.saturation * (rgb - luma)
    rgb = (rgb - 0.5) * profile.contrast + 0.5
    rgb = rgb + profile.brightness
    rgb = np.clip(rgb, 0.0, 1.0)
    return np.round(rgb * 255.0).astype(np.uint8)


def profile_for_client(client_id: int) -> StainProfile:
    return PROFILES[client_id % len(PROFILES)]
