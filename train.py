# =============================================================================
# train.py — Final (paper-faithful Qm integration)
# =============================================================================

import numpy as np
from config import CFG
from environment import Environment
from metrics import Metrics
from priority import Priority
from reward import Reward
from agent import Agent


def train(seed=42, mode="pa", beta=0.5, use_qm=False):

    rng = np.random.default_rng(seed)

    env     = Environment(rng)
    metrics = Metrics(env, rng)
    prio    = Priority(env)
    reward  = Reward(metrics)

    agent = Agent(rng, beta=beta, use_qm=use_qm)

    eps = CFG.EPS_START

    log = dict(reward=[], qed=[], scr=[], cdp=[], noff=[], eps=[])

    for g in range(CFG.G_MAX):

        agent.reset_episode()

        energy = rng.uniform(0.3 * CFG.E_MAX, CFG.E_MAX, CFG.N)
        alive  = energy > 0
        avail  = list(range(CFG.N))
        state  = CFG.N

        total_r = 0.0
        pri = prio.compute(alive, energy)

        for _ in range(CFG.N):

            act = agent.act(state, avail, pri, eps, mode)

            energy[act] = CFG.E_MAX
            energy -= env.vcs
            energy = np.maximum(0.0, energy)

            alive = energy > 0
            pri   = prio.compute(alive, energy)

            r = reward.compute(alive, energy, act, pri, env, mode)

            nxt = [a for a in avail if a != act]

            agent.update(state, act, r, act, nxt)

            total_r += r
            avail = nxt
            state = act

        # 🔥 Qm update (IMPORTANT)
        agent.update_qm_from_episode(mode, total_r)

        q_, s_, c_ = metrics.qed(alive)

        eps *= CFG.EPS_DECAY

        log["reward"].append(total_r)
        log["qed"].append(q_)
        log["scr"].append(s_)
        log["cdp"].append(c_)
        log["noff"].append(int((energy == 0).sum()))
        log["eps"].append(eps)

    log["_env"]  = env
    log["_prio"] = prio

    return log