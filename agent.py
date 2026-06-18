# =============================================================================
# agent.py — Paper-faithful MEQA / PA-MEQA with Qm-table
# =============================================================================

import numpy as np
from config import CFG


class Agent:

    def __init__(self, rng, beta=0.5, use_qm=True):
        self.rng = rng
        self.beta = float(np.clip(beta, 0.0, 1.0))
        self.use_qm = use_qm

        self.Q  = np.zeros((CFG.N + 1, CFG.N + 1))
        self.Qm = np.full((CFG.N + 1, CFG.N + 1), -np.inf)

        self._episode_sa = []

    # =========================
    def reset_episode(self):
        self._episode_sa = []

    # =========================
    def act(self, state, avail, pri, eps, mode="pa"):

        # ---- Exploration ----
        if self.rng.random() <= eps:
            if mode == "pa":
                logits = self.Q[state, avail] + 0.25 * pri[avail]
                logits -= logits.max()
                probs = np.exp(logits)
                probs /= probs.sum()
                return int(self.rng.choice(avail, p=probs))
            return int(self.rng.choice(avail))

        # ---- QL ----
        if mode == "ql" or not self.use_qm:
            return avail[int(np.argmax(self.Q[state, avail]))]

        # ---- MEQA / PA ----
        q_vals  = self.Q[state, avail]
        qm_vals = self.Qm[state, avail]

        qm_safe = np.where(np.isfinite(qm_vals), qm_vals, q_vals)

        blended = (1 - self.beta) * q_vals + self.beta * qm_safe

        if mode == "pa":
            blended = blended + 0.25 * pri[avail]

        return avail[int(np.argmax(blended))]

    # =========================
    def update(self, s, a, r, s_next, avail_next):

        q_next = self.Q[s_next, avail_next].max() if avail_next else 0.0
        target = r + CFG.GAMMA * q_next

        self.Q[s, a] += CFG.ALPHA * (target - self.Q[s, a])

        self._episode_sa.append((s, a))

    # =========================
    def update_qm_from_episode(self, mode, total_reward):

        if mode == "ql" or not self.use_qm:
            self._episode_sa = []
            return

        for (s, a) in self._episode_sa:
            candidate = max(self.Q[s, a], total_reward)

            if candidate > self.Qm[s, a]:
                self.Qm[s, a] = candidate

        self._episode_sa = []