"""Coordinate masks for sparse role-based PSO variants.

Sparse role PSO applies role-based movement modifications only to selected
coordinates of the velocity update, analogously to gene-level mutation in
evolutionary algorithms.
"""

from __future__ import annotations

import numpy as np


class CoordinateMaskMixin:
    """Reusable helper for coordinate-wise sparse role behavior."""

    def coordinate_count(
            self,
            dim: int,
            mode: str,
            fraction: float = 0.1,
            scale: float = 1.0,
            count: int = 10,
    ) -> int:
        """Return the exact number of selected coordinates for a mask.

        Supported modes:
        - ``fraction``: ``int(dim * fraction)``
        - ``sqrt``: ``int(scale * sqrt(dim))``
        - ``log``: ``int(scale * log(dim))``
        - ``constant``: ``int(count)``

        The result is always clamped to ``1 <= result <= dim``.
        """
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")

        if mode == "fraction":
            selected = int(dim * fraction)
        elif mode == "sqrt":
            selected = int(scale * np.sqrt(dim))
        elif mode == "log":
            selected = int(scale * np.log(dim))
        elif mode == "constant":
            selected = int(count)
        else:
            raise ValueError(f"Invalid coordinate mask mode: {mode}")

        return max(1, min(dim, selected))

    def coordinate_mask(
            self,
            dim: int,
            mode: str = "sqrt",
            fraction: float = 0.1,
            scale: float = 1.0,
            count: int = 10,
    ) -> np.ndarray:
        """Return a boolean mask selecting exactly ``count`` coordinates."""
        selected = self.coordinate_count(
            dim=dim,
            mode=mode,
            fraction=fraction,
            scale=scale,
            count=count,
        )
        return self.coordinate_mask_by_count(dim=dim, count=selected)

    @staticmethod
    def coordinate_mask_by_count(dim: int, count: int) -> np.ndarray:
        """Return a boolean mask of length ``dim`` with exactly ``count`` true values."""
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")

        selected = max(1, min(dim, int(count)))
        mask = np.zeros(dim, dtype=bool)
        indices = np.random.choice(dim, size=selected, replace=False)
        mask[indices] = True
        return mask

    @staticmethod
    def mix_by_mask(
            normal_vec: np.ndarray,
            role_vec: np.ndarray,
            mask: np.ndarray,
    ) -> np.ndarray:
        """Use ``role_vec`` on masked coordinates and ``normal_vec`` elsewhere."""
        return np.where(mask, role_vec, normal_vec)
