# =============================================================================
# environment.py — WRSN environment: sensor layout, event grid, detection model
# =============================================================================

import numpy as np
from config import CFG


class Environment:
    """
    Represents the physical WRSN environment.

    Attributes
    ----------
    xy   : (N, 2)  — sensor node (x, y) coordinates
    vcs  : (N,)    — per-node energy consumption rate (J/s)
    locs : (T², 2) — candidate event locations on the T×T grid
    pdeo : (T²,)   — Probability Distribution of Event Occurrence (normalised)
    P    : (N, T²) — PSM detection probability matrix
    """

    def __init__(self, rng: np.random.Generator):
        self.rng = rng

        self.xy  = rng.uniform(0, CFG.L, (CFG.N, 2))
        self.vcs = rng.uniform(1, 3, CFG.N)

        xs, ys    = np.meshgrid(
            np.linspace(0, CFG.L, CFG.T),
            np.linspace(0, CFG.L, CFG.T),
        )
        self.locs = np.stack([xs.ravel(), ys.ravel()], axis=1)  # (T², 2)

        self.pdeo = self._pdeo()
        self.P    = self._detection_matrix()

    def _pdeo(self) -> np.ndarray:
        """3–5 Gaussian hotspots + uniform background, normalised to sum=1."""
        n  = self.rng.integers(3, 6)
        cx = self.rng.uniform(0.2 * CFG.L, 0.8 * CFG.L, n)
        cy = self.rng.uniform(0.2 * CFG.L, 0.8 * CFG.L, n)
        p  = sum(
            np.exp(
                -((self.locs[:, 0] - hx) ** 2 + (self.locs[:, 1] - hy) ** 2)
                / (2 * (0.15 * CFG.L) ** 2)
            )
            for hx, hy in zip(cx, cy)
        ) + 0.05
        return p / p.sum()

    def _detection_matrix(self) -> np.ndarray:
        """
        PSM detection probability  P[i,j]  for node i and event location j.
            = 1                              if d ≤ r_g
            = exp(−δ·(d−r_g)^β)             if r_g < d ≤ r_p
            = 0                              otherwise
        """
        d = np.linalg.norm(self.xy[:, None] - self.locs[None], axis=-1)
        return np.where(
            d <= CFG.RG, 1.0,
            np.where(d <= CFG.RP,
                     np.exp(-CFG.DELTA * (d - CFG.RG) ** CFG.BETA),
                     0.0),
        )
