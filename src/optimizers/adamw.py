import torch


class AdamW:
    def __init__(self, lr: float, weight_decay: float, **_ignored):
        self.lr = lr
        self.weight_decay = weight_decay

    def __call__(self, params):
        return torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
