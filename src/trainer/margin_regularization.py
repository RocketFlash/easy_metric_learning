from ..utils import AverageMeter


def get_margin_module(model, accelerator=None):
    if accelerator is not None:
        model = accelerator.unwrap_model(model)
    return getattr(model, "margin", None)


def add_margin_regularization_loss(
    total_loss,
    loss_meters,
    model,
    accelerator=None,
    update_meter=True,
):
    margin = get_margin_module(model, accelerator=accelerator)
    if margin is None or not hasattr(margin, "regularization_loss"):
        return total_loss

    regularization_loss = margin.regularization_loss()
    if regularization_loss is None:
        return total_loss

    if update_meter:
        loss_name = getattr(margin, "regularization_loss_name", "margin_reg")
        loss_meters.setdefault(loss_name, AverageMeter())
        loss_meters[loss_name].update(regularization_loss.detach().item())

    return total_loss + regularization_loss
