class LBFGS:
    """Config holder for pykan's built-in LBFGS (not a usable optimizer factory)."""

    PYKAN_OPT = "LBFGS"

    def __init__(self, lr: float = 1.0, **_ignored):
        self.lr = lr
