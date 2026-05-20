import pytest

torch = pytest.importorskip("torch")

from src.model.model import EmbeddingsNet, EmbeddigsNet


def test_embeddings_net_typo_alias_preserves_import_compatibility():
    assert EmbeddigsNet is EmbeddingsNet
