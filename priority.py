# =============================================================================
# priority.py — Node priority scoring (used only by PA-MEQA)
# =============================================================================

import numpy as np
from config      import CFG
from environment import Environment


class Priority:

    def __init__(self, env: Environment):
        self.env = env

    def compute(self, alive: np.ndarray, energy: np.ndarray) -> np.ndarray:
        det       = (self.env.P * self.env.pdeo[None]).sum(axis=1)
        urg       = 1.0 - energy / CFG.E_MAX
        ttf       = energy / (self.env.vcs + 1e-9)
        ttf_score = 1.0 / (ttf + 1e-6)

        def _norm(x):
            return x / (x.max() + 1e-9)

        pri          = 0.4 * _norm(det) + 0.3 * _norm(ttf_score) + 0.3 * _norm(urg)
        pri[~alive]  = 0.0
        return pri / (pri.max() + 1e-9)
