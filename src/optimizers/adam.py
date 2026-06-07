import torch


class Adam:
    PYKAN_OPT = "Adam"

    def __init__(self, lr: float, **_ignored):
        self.lr = lr

    def __call__(self, params):
        return torch.optim.Adam(params, lr=self.lr)
