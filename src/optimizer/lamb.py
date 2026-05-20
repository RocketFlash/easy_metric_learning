import torch
from torch.optim import Optimizer


class LAMB(Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-6,
        weight_decay=0.0,
        min_trust=0.0,
        max_trust=10.0,
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            min_trust=min_trust,
            max_trust=max_trust,
        )
        super(LAMB, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                grad = parameter.grad
                if grad.is_sparse:
                    raise RuntimeError("LAMB does not support sparse gradients")

                state = self.state[parameter]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(parameter)
                    state["exp_avg_sq"] = torch.zeros_like(parameter)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                state["step"] += 1

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]
                adam_step = exp_avg / bias_correction1
                denom = (exp_avg_sq / bias_correction2).sqrt().add_(group["eps"])
                adam_step = adam_step / denom

                if group["weight_decay"] != 0:
                    adam_step = adam_step.add(parameter, alpha=group["weight_decay"])

                weight_norm = parameter.norm(p=2)
                update_norm = adam_step.norm(p=2)
                if weight_norm > 0 and update_norm > 0:
                    trust_ratio = weight_norm / update_norm
                    trust_ratio = trust_ratio.clamp(
                        group["min_trust"], group["max_trust"]
                    )
                else:
                    trust_ratio = torch.ones((), device=parameter.device)

                parameter.add_(adam_step * trust_ratio, alpha=-group["lr"])

        return loss
