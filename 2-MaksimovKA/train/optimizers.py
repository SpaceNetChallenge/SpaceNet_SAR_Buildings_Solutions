import torch
from radam import RAdam
from Adam import Adam_GCC


def get_optimizer(optimizer_name, model, lr, momentum, decay):
    if optimizer_name == 'radam':
        
        optimizer = RAdam(model.parameters(),
                          lr,
                          weight_decay=decay)
    elif optimizer_name == 'adam':
        
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr,
                                     weight_decay=decay)
        
    elif optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr,
                                    momentum=momentum,
                                    weight_decay=decay)
    elif optimizer_name == 'adam_gcc':
        print('ADAM_GCC optimizer')
        optimizer = Adam_GCC(model.parameters(),
                             lr,
                             weight_decay=decay)
    
    else:
        optimizer = None
    return optimizer
