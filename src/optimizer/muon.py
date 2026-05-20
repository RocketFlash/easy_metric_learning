import torch
from torch.optim import Optimizer


def zeropower_via_newtonschulz(matrix, steps=5, eps=1e-7):
    if matrix.ndim != 2:
        return matrix

    transposed = matrix.size(0) > matrix.size(1)
    if transposed:
        matrix = matrix.t()

    matrix = matrix / (matrix.norm() + eps)
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(steps):
        gram = matrix @ matrix.t()
        matrix = a * matrix + (b * gram + c * gram @ gram) @ matrix

    if transposed:
        matrix = matrix.t()
    return matrix


class Muon(Optimizer):
    """
    Muon-style optimizer for matrix parameters with AdamW fallback for vectors.
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        momentum=0.95,
        weight_decay=0.0,
        ns_steps=5,
        adam_betas=(0.9, 0.999),
        adam_eps=1e-8,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            ns_steps=ns_steps,
            adam_betas=adam_betas,
            adam_eps=adam_eps,
        )
        super(Muon, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["adam_betas"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                grad = parameter.grad
                state = self.state[parameter]
                if len(state) == 0:
                    state["step"] = 0
                    state["momentum_buffer"] = torch.zeros_like(parameter)
                    state["exp_avg"] = torch.zeros_like(parameter)
                    state["exp_avg_sq"] = torch.zeros_like(parameter)

                state["step"] += 1
                if parameter.ndim >= 2:
                    buffer = state["momentum_buffer"]
                    buffer.mul_(group["momentum"]).add_(grad)
                    update = zeropower_via_newtonschulz(
                        buffer.view(buffer.size(0), -1),
                        steps=group["ns_steps"],
                    ).view_as(parameter)
                    if group["weight_decay"] != 0:
                        parameter.mul_(1 - group["lr"] * group["weight_decay"])
                    parameter.add_(update, alpha=-group["lr"])
                else:
                    exp_avg = state["exp_avg"]
                    exp_avg_sq = state["exp_avg_sq"]
                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    bias_correction1 = 1 - beta1 ** state["step"]
                    bias_correction2 = 1 - beta2 ** state["step"]
                    step_size = group["lr"] / bias_correction1
                    denom = (
                        (exp_avg_sq / bias_correction2).sqrt().add_(group["adam_eps"])
                    )
                    if group["weight_decay"] != 0:
                        parameter.mul_(1 - group["lr"] * group["weight_decay"])
                    parameter.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
