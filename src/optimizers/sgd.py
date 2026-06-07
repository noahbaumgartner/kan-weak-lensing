import torch


class SGD:
    def __init__(self, lr: float, momentum: float, weight_decay: float, **_ignored):
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay

    def __call__(self, params):
        return torch.optim.SGD(
            params,
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )
