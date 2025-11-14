import torch.optim as optim


def get_optimizer(
        model,
        type="Adam",
        lr=5e-4,
):
    if type == "Adam":
        return optim.Adam(model.parameters(), lr=lr)
    else:
        raise TypeError(f"Unknown optimizer_type: {type}")
