import torch.optim as optim

OPTIMIZERS = {
    "sgd": (optim.SGD, {"lr": 0.01, "momentum": 0.9}),
    "adam": (optim.Adam, {"lr": 1e-3}),
}
