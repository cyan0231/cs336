import math
def get_lr(train_step,max_lr,min_lr,warmup_iter,cosine_cycle_iter):
    if train_step<=warmup_iter:
        return max_lr*train_step/warmup_iter
    if train_step>=cosine_cycle_iter:
        return min_lr
    decay_rate=(train_step-warmup_iter)/(cosine_cycle_iter-warmup_iter)
    coeff=0.5*(1.0+math.cos(math.pi*decay_rate))
    return min_lr+coeff*(max_lr-min_lr)
