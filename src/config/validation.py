PAIR_BATCH_LOSSES = {
    "circle",
    "multi_similarity",
    "batch_hard_triplet",
    "supcon",
    "ntxent",
    "unitsface",
}
POSITIVE_PAIR_SAMPLERS = {"balanced", "pk", "hard_negative", "hierarchical_pk"}
REFRESHABLE_HARD_NEGATIVE_SAMPLERS = {"hard_negative"}


class ConfigValidationError(ValueError):
    pass


def _value(config, key, default=None):
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def _loss_names(config):
    losses = _value(_value(config, "loss", None), "losses", [])
    return {_value(loss_cfg, "name") for loss_cfg in losses}


def validate_training_config(config, raise_on_error=True):
    errors = []

    ddp = bool(_value(config, "ddp", False))
    trainer = _value(_value(config, "train", None), "trainer", None)
    sampler = _value(_value(config, "dataloader", None), "sampler", None)
    sampler_type = _value(sampler, "type", "default")
    fsdp = _value(trainer, "fsdp", None)
    compile_cfg = _value(trainer, "compile", None)
    if _value(fsdp, "enabled", False) and not ddp:
        errors.append("train.trainer.fsdp.enabled requires ddp=True")
    if _value(fsdp, "enabled", False) and _value(compile_cfg, "enabled", False):
        errors.append("FSDP and torch.compile should not be enabled together")
    hard_negative_cache = _value(trainer, "hard_negative_cache", None)
    xbm = _value(trainer, "xbm", None)
    if _value(xbm, "enabled", False):
        memory_size = _value(xbm, "memory_size", 4096)
        if memory_size is None or int(memory_size) <= 0:
            errors.append("xbm.memory_size must be positive")

    if _value(hard_negative_cache, "enabled", False):
        refresh_every = _value(hard_negative_cache, "refresh_every", 1)
        if refresh_every is None or int(refresh_every) <= 0:
            errors.append("hard_negative_cache.refresh_every must be positive")
        max_batches = _value(hard_negative_cache, "max_batches", None)
        if max_batches is not None and int(max_batches) <= 0:
            errors.append("hard_negative_cache.max_batches must be positive")
        if (
            _value(hard_negative_cache, "update_sampler", True)
            and sampler_type not in REFRESHABLE_HARD_NEGATIVE_SAMPLERS
        ):
            errors.append(
                "hard_negative_cache.update_sampler=True requires "
                "dataloader=hard_negative; set update_sampler=False to only write "
                "the cache"
            )

    loss_names = _loss_names(config)
    if loss_names & PAIR_BATCH_LOSSES and sampler_type not in POSITIVE_PAIR_SAMPLERS:
        errors.append(
            "pair/batch losses require a positive-pair sampler such as dataloader=pk"
        )
    if sampler_type == "hierarchical_pk":
        dataset = _value(config, "dataset", None)
        if (
            _value(dataset, "group_column", None) is None
            and _value(sampler, "groups", None) is None
        ):
            errors.append(
                "hierarchical_pk requires dataset.group_column or sampler.groups"
            )
    if (
        sampler_type == "hard_negative"
        and _value(sampler, "embeddings", None) is None
        and _value(sampler, "embeddings_path", None) is None
    ):
        errors.append("hard_negative sampler requires embeddings_path or embeddings")

    evaluator = _value(_value(config, "evaluation", None), "evaluator", None)
    knn = _value(_value(config, "evaluation", None), "knn", None)
    rerank = _value(knn, "rerank", None)
    if _value(rerank, "enabled", False):
        k1 = _value(rerank, "k1", 20)
        k2 = _value(rerank, "k2", 6)
        lambda_value = _value(rerank, "lambda_value", 0.3)
        if k1 is None or int(k1) <= 0:
            errors.append("evaluation.knn.rerank.k1 must be positive")
        if k2 is None or int(k2) <= 0:
            errors.append("evaluation.knn.rerank.k2 must be positive")
        if lambda_value is None or not 0 <= float(lambda_value) <= 1:
            errors.append("evaluation.knn.rerank.lambda_value must be in [0, 1]")

    evaluator_type = _value(evaluator, "type", "base")
    if evaluator_type == "ijb_template":
        if _value(evaluator, "metadata_path", None) is None:
            errors.append("ijb_template evaluator requires metadata_path")
        if _value(evaluator, "pairs_path", None) is None:
            errors.append("ijb_template evaluator requires pairs_path")
    if evaluator_type == "face_verification" and _value(evaluator, "pairs_path", None):
        valid_protocols = {"csv", "lfw", "agedb", "cfp", "whitespace"}
        protocol = _value(evaluator, "pair_protocol", "csv")
        if protocol not in valid_protocols:
            errors.append(f"Unknown face verification pair_protocol: {protocol}")

    if errors and raise_on_error:
        raise ConfigValidationError("; ".join(errors))
    return errors
