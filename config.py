class CFG:
    # ── Network ───────────────────────────────────────────────────────────────
    N          = 60       # Number of sensor nodes
    L          = 100      # Length of monitoring area (m)
    E_MAX      = 144.0    # Maximum energy of Mobile Charger (kJ)

    # ── Sensing model (PSM) ───────────────────────────────────────────────────
    RG         = 2.0      # Guaranteed sensing radius (m)
    RP         = 4.0      # Probability sensing radius (m)
    DELTA      = 0.2      # Detection attenuation rate (δ)
    BETA       = 2.0      # Nonlinearity coefficient (β)

    # ── Event grid ────────────────────────────────────────────────────────────
    T          = 20       # Grid resolution: T×T event locations

    # ── Training ──────────────────────────────────────────────────────────────
    G_MAX      = 2000      # Total training iterations per run
    N_RUNS     = 3        # Number of independent experiment runs
    SEED       = 42       # Base random seed

    # ── Q-learning ────────────────────────────────────────────────────────────
    ALPHA      = 0.3      # Learning rate (α)
    GAMMA      = 0.98     # Discount factor (γ)

    # ── Exploration ───────────────────────────────────────────────────────────
    EPS_START  = 0.9      # Initial ε
    EPS_DECAY  = 0.995    # Per-iteration ε decay multiplier

    # ── Reward weights ────────────────────────────────────────────────────────
    W1         = 0.3      # Weight for SCR
    W2         = 0.7      # Weight for CDP

    # ── Priority (PA-MEQA only) ───────────────────────────────────────────────
    LAM_BASE   = 0.2      # Base priority scaling factor (λ)

    # ── Monte Carlo coverage estimation ───────────────────────────────────────
    MC_SAMPLES = 200      # Sample points for TSA approximation
