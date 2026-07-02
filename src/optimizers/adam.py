import torch


class Adam:
    def __init__(self, lr: float, weight_decay: float = 0.0, **_ignored):
        self.lr = lr
        self.weight_decay = weight_decay

    def __call__(self, params):
        return torch.optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)
