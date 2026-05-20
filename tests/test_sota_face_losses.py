import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from src.evaluator.quality import qmagface_pair_scores
from src.loss.boundaryface import BoundaryFaceLoss
from src.loss.sface import SFaceLoss
from src.loss.unitsface import UniTSFaceLoss


@pytest.mark.parametrize(
    "loss_fn",
    [
        SFaceLoss(in_features=8, out_features=3),
        BoundaryFaceLoss(in_features=8, out_features=3),
        UniTSFaceLoss(),
    ],
)
def test_sota_face_losses_have_gradients(loss_fn):
    embeddings = torch.randn(6, 8, requires_grad=True)
    labels = torch.tensor([0, 0, 1, 1, 2, 2])

    loss = loss_fn(embeddings, labels)
    loss.backward()

    assert loss.ndim == 0
    assert embeddings.grad is not None


def test_boundaryface_accepts_non_long_labels():
    loss_fn = BoundaryFaceLoss(in_features=8, out_features=3)
    embeddings = torch.randn(3, 8, requires_grad=True)
    labels = torch.tensor([0, 1, 2], dtype=torch.int32)

    loss = loss_fn(embeddings, labels)

    assert loss.ndim == 0


def test_qmagface_pair_scores_boost_high_quality_pairs():
    embeddings = np.array(
        [
            [2.0, 0.0],
            [2.0, 0.0],
            [0.5, 0.0],
            [0.5, 0.0],
        ],
        dtype=np.float32,
    )
    pairs = np.array([[0, 1], [2, 3]])

    scores = qmagface_pair_scores(embeddings, pairs, alpha=0.2)

    assert scores[0] > scores[1]
