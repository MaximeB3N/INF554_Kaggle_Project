def get_lr(optimizer):
    """Get the current learning rate from optimizer"""
    for param_group in optimizer.param_groups:
        return param_group['lr']