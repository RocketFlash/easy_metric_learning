import torch
from torch.optim import Optimizer


class ScheduleFreeAdamW(Optimizer):
    """
    AdamW with schedule-free iterate averaging.

    The optimizer maintains an online average of train iterates and exposes
    train()/eval() methods for swapping between the live and averaged weights.
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        averaging=0.9,
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            averaging=averaging,
        )
        super(ScheduleFreeAdamW, self).__init__(params, defaults)
        self._eval_mode = False

    @torch.no_grad()
    def train(self):
        if not self._eval_mode:
            return
        for group in self.param_groups:
            for parameter in group["params"]:
                state = self.state.get(parameter, None)
                if state is not None and "train_parameter" in state:
                    parameter.copy_(state.pop("train_parameter"))
        self._eval_mode = False

    @torch.no_grad()
    def eval(self):
        if self._eval_mode:
            return
        for group in self.param_groups:
            for parameter in group["params"]:
                state = self.state.get(parameter, None)
                if state is not None and "weight_average" in state:
                    state["train_parameter"] = parameter.detach().clone()
                    parameter.copy_(state["weight_average"])
        self._eval_mode = True

    def state_dict(self):
        if self._eval_mode:
            self.train()
        return super().state_dict()

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self._eval_mode = False

    @torch.no_grad()
    def step(self, closure=None):
        if self._eval_mode:
            self.train()

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
                    raise RuntimeError(
                        "ScheduleFreeAdamW does not support sparse grads"
                    )

                state = self.state[parameter]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(parameter)
                    state["exp_avg_sq"] = torch.zeros_like(parameter)
                    state["weight_average"] = parameter.detach().clone()

                state["step"] += 1
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]
                denom = (exp_avg_sq / bias_correction2).sqrt().add_(group["eps"])

                if group["weight_decay"] != 0:
                    parameter.mul_(1 - group["lr"] * group["weight_decay"])
                parameter.addcdiv_(
                    exp_avg,
                    denom,
                    value=-group["lr"] / bias_correction1,
                )

                average = state["weight_average"]
                average.mul_(group["averaging"]).add_(
                    parameter, alpha=1 - group["averaging"]
                )

        return loss
