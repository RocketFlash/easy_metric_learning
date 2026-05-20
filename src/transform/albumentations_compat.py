import inspect

import albumentations as A


def coarse_dropout(
    num_holes_range=(1, 2),
    hole_height_range=(8, 8),
    hole_width_range=(8, 8),
    fill=0,
    fill_mask=None,
    p=0.5,
):
    signature = inspect.signature(A.CoarseDropout)
    if "num_holes_range" in signature.parameters:
        return A.CoarseDropout(
            num_holes_range=tuple(num_holes_range),
            hole_height_range=tuple(hole_height_range),
            hole_width_range=tuple(hole_width_range),
            fill=fill,
            fill_mask=fill_mask,
            p=p,
        )

    return A.CoarseDropout(
        min_holes=int(num_holes_range[0]),
        max_holes=int(num_holes_range[1]),
        min_height=int(hole_height_range[0]),
        max_height=int(hole_height_range[1]),
        min_width=int(hole_width_range[0]),
        max_width=int(hole_width_range[1]),
        fill_value=fill,
        mask_fill_value=fill_mask,
        p=p,
    )
