# =============================================================================
# reward.py — ORIGINAL PA-MEQA reward (paper version)
# =============================================================================

import numpy as np
from config import CFG
from metrics import Metrics
from environment import Environment


class Reward:

    def __init__(self, metrics: Metrics):
        self.m = metrics

    def compute(self, alive, energy, act, pri, env, mode="pa"):

        q, s, c = self.m.qed(alive)

        # ---- QL ----
        if mode == "ql":
            return energy[act] / CFG.E_MAX

        # ---- MEQA ----
        if mode == "meqa":
            return q

        # ---- PA-MEQA (ORIGINAL PAPER) ----
        lam = CFG.LAM_BASE * (1 - alive.mean())

        future_dead = ((energy - env.vcs * 5) <= 0).mean()
        redundancy  = (env.P[alive].sum(axis=0) > 2).mean()
        dead_ratio  = 1.0 - alive.mean()

        return (
            q
            + lam * pri[act]
            - 0.10 * future_dead
            - 0.08 * redundancy
            - 0.12 * dead_ratio
        )