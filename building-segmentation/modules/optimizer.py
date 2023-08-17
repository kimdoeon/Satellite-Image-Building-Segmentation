import torch.optim as optim

def get_optimizer(name: str):
    if name == 'SGD':
        return optim.SGD

    elif name == 'Adam':
        return optim.Adam
    
    elif name == 'AdamW':
        return optim.AdamW
