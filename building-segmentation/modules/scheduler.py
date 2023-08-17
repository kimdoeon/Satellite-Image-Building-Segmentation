from torch.optim import lr_scheduler

def get_scheduler(name):
    if name == 'CosineAnnealingLR':
        return lr_scheduler.CosineAnnealingLR

    elif name == 'CosineAnnealingWarmRestarts':
        return lr_scheduler.CosineAnnealingWarmRestarts

    elif name == 'PolynomialLR':
        return lr_scheduler.PolynomialLR

    elif name == 'ExponentialLR':
        return lr_scheduler.ExponentialLR

    else:
        return None
    