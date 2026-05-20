import torch

EMBEDDING_INPUT_KEYS = {"embedding", "embeddings", "emb"}


def _config_value(config, key, default=None):
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def get_xbm_config(config):
    trainer_config = _config_value(
        _config_value(config, "train", None), "trainer", None
    )
    return _config_value(trainer_config, "xbm", None)


def create_xbm(config, device=None):
    xbm_config = get_xbm_config(config)
    if not _config_value(xbm_config, "enabled", False):
        return None

    return CrossBatchMemory(
        memory_size=int(_config_value(xbm_config, "memory_size", 4096)),
        embedding_size=int(_config_value(config, "embeddings_size")),
        device=device,
    )


def is_xbm_target_compatible(targets):
    return not isinstance(targets, (list, tuple))


def get_xbm_for_targets(xbm, targets):
    if xbm is None or not is_xbm_target_compatible(targets):
        return None
    return xbm


def update_xbm(xbm, embeddings, targets):
    if xbm is not None and is_xbm_target_compatible(targets):
        xbm.enqueue(embeddings, targets)


def is_xbm_compatible_loss(loss_params):
    input_key = getattr(loss_params, "input", "output")
    return getattr(loss_params, "xbm", True) and input_key in EMBEDDING_INPUT_KEYS


class CrossBatchMemory:
    def __init__(self, memory_size, embedding_size, device=None):
        if int(memory_size) <= 0:
            raise ValueError("xbm.memory_size must be positive")
        if int(embedding_size) <= 0:
            raise ValueError("xbm embedding_size must be positive")

        self.memory_size = int(memory_size)
        self.embedding_size = int(embedding_size)
        self.device = torch.device(device) if device is not None else None
        self.embeddings = None
        self.labels = None
        self.ptr = 0
        self.full = False

    def __len__(self):
        if self.embeddings is None:
            return 0
        return self.memory_size if self.full else self.ptr

    def _ensure_storage(self, embeddings):
        device = self.device if self.device is not None else embeddings.device
        if (
            self.embeddings is not None
            and self.embeddings.device == device
            and self.embeddings.dtype == embeddings.dtype
        ):
            return

        self.embeddings = torch.zeros(
            self.memory_size,
            self.embedding_size,
            device=device,
            dtype=embeddings.dtype,
        )
        self.labels = torch.zeros(self.memory_size, device=device, dtype=torch.long)
        self.ptr = 0
        self.full = False

    def _indices(self):
        if len(self) == 0:
            return None
        if not self.full:
            return torch.arange(self.ptr, device=self.embeddings.device)
        return torch.cat(
            [
                torch.arange(self.ptr, self.memory_size, device=self.embeddings.device),
                torch.arange(0, self.ptr, device=self.embeddings.device),
            ]
        )

    def get(self, embeddings):
        if len(self) == 0:
            empty_embeddings = embeddings.new_empty((0, embeddings.size(1)))
            empty_labels = torch.empty(0, device=embeddings.device, dtype=torch.long)
            return empty_embeddings, empty_labels

        indices = self._indices()
        memory_embeddings = self.embeddings[indices].to(
            device=embeddings.device,
            dtype=embeddings.dtype,
        )
        memory_labels = self.labels[indices].to(device=embeddings.device)
        return memory_embeddings, memory_labels

    def extend(self, embeddings, labels):
        memory_embeddings, memory_labels = self.get(embeddings)
        if memory_embeddings.numel() == 0:
            return embeddings, labels
        labels = labels.view(-1).long().to(device=embeddings.device)
        return (
            torch.cat([embeddings, memory_embeddings], dim=0),
            torch.cat([labels, memory_labels], dim=0),
        )

    def enqueue(self, embeddings, labels):
        if embeddings.ndim != 2:
            raise ValueError("XBM expects 2D embeddings")
        if embeddings.size(1) != self.embedding_size:
            raise ValueError(
                f"XBM embedding size mismatch: expected {self.embedding_size}, "
                f"got {embeddings.size(1)}"
            )

        self._ensure_storage(embeddings)
        embeddings = embeddings.detach().to(self.embeddings.device)
        labels = labels.detach().view(-1).long().to(self.labels.device)
        if embeddings.size(0) != labels.size(0):
            raise ValueError("XBM embeddings and labels must have the same length")

        if embeddings.size(0) >= self.memory_size:
            self.embeddings.copy_(embeddings[-self.memory_size :])
            self.labels.copy_(labels[-self.memory_size :])
            self.ptr = 0
            self.full = True
            return

        remaining = self.memory_size - self.ptr
        first = min(remaining, embeddings.size(0))
        second = embeddings.size(0) - first

        self.embeddings[self.ptr : self.ptr + first] = embeddings[:first]
        self.labels[self.ptr : self.ptr + first] = labels[:first]
        if second > 0:
            self.embeddings[:second] = embeddings[first:]
            self.labels[:second] = labels[first:]
            self.full = True

        self.ptr = (self.ptr + embeddings.size(0)) % self.memory_size
        if self.ptr == 0:
            self.full = True
