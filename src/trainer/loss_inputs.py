from .xbm import is_xbm_compatible_loss


def get_loss_input(loss_params, output, embeddings, images=None):
    input_key = getattr(loss_params, "input", "output")
    if input_key in {"embedding", "embeddings", "emb"}:
        return embeddings
    if input_key in {
        "image_embeddings",
        "images_embeddings",
        "input_embeddings",
        "inputs_embeddings",
    }:
        if images is None:
            raise ValueError(f"Loss input {input_key} requires images")
        return images, embeddings
    return output


def calculate_weighted_loss(
    loss_params,
    output,
    embeddings,
    targets,
    images=None,
    xbm=None,
):
    loss_input = get_loss_input(loss_params, output, embeddings, images=images)
    if (
        xbm is not None
        and is_xbm_compatible_loss(loss_params)
        and not isinstance(loss_input, tuple)
    ):
        loss_input, targets = xbm.extend(loss_input, targets)
    if isinstance(loss_input, tuple):
        return loss_params.loss_fn(*loss_input, targets) * loss_params.weight
    return loss_params.loss_fn(loss_input, targets) * loss_params.weight
