import os
import torch

@torch.compile
def zeropower_via_newtonschulz5(G, steps=10, eps=1e-7):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    X /= (X.norm() + eps) # ensure top singular value <= 1
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A # adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X

class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - This optimizer assumes that the scale of the learning rate is decoupled from the update norm.
    As a result, modifying the learning rate schedule has a very different effect than in Adam.
    For this reason, this optimizer should be run with a schedule where the learning rate decreases
    linearly to zero, and the number of steps at peak learning rate should be carefully tuned.
    - This optimizer is not designed to be used for parameters with fewer than 2 dimensions.
    """
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            zero_power_via_newtonschulz5 = zeropower_via_newtonschulz5
            if 'ns_steps' in group:
                def zero_power_via_newtonschulz5(G):
                    return zeropower_via_newtonschulz5(G, steps=group['ns_steps'])
            
            for p in group['params']:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g)
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(g)
                if group['nesterov']:
                    g = g.add(buf, alpha=momentum)
                else:
                    g = buf
                
                # Apply orthogonalization
                # [Fix] 适配维度大于 2 的张量情况 (如 Embedding 或 Conv 结构)
                orig_shape = g.shape
                if len(orig_shape) > 2:
                    g = g.view(orig_shape[0], -1)
                    
                g = zero_power_via_newtonschulz5(g).type_as(p)
                
                if len(orig_shape) > 2:
                    g = g.view(orig_shape)
                    
                p.data.add_(g, alpha=-lr * max(1, g.size(0)/g.size(1))**0.5 if len(orig_shape) == 2 else -lr)
