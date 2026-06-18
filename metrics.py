import numpy as np
from config      import CFG
from environment import Environment


class Metrics:

    def __init__(self, env: Environment, rng: np.random.Generator):
        self.env     = env
        self.rng     = rng
        self.cdp_max = max(self._cdp(np.ones(CFG.N, bool)), 1e-9)

    def _cdp(self, alive: np.ndarray) -> float:
        if not alive.any():
            return 0.0
        return float(np.dot(
            1.0 - np.prod(1.0 - self.env.P[alive], axis=0),
            self.env.pdeo,
        ))

    def _scr(self, alive: np.ndarray) -> float:
        if not alive.any():
            return 0.0
        pts = self.rng.uniform(0, CFG.L, (CFG.MC_SAMPLES, 2))
        d   = np.linalg.norm(pts[:, None] - self.env.xy[None, alive], axis=-1)
        return float((d <= CFG.RP).any(axis=1).mean())

    def qed(self, alive: np.ndarray):
        """Returns (QED, SCR, CDP)."""
        s = self._scr(alive)
        c = self._cdp(alive)
        c1=(c / self.cdp_max)
        q = CFG.W1 * s + CFG.W2 * c1
        return q, s, c1
