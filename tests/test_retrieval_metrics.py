import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")

from src.metric.basic import mean_average_precision_at_r, nearest_label_nmi


def test_mean_average_precision_at_r_uses_class_cardinality_minus_query():
    y_true = np.array(["a", "a", "a", "b", "b"], dtype=object)
    y_pred = np.array(
        [
            ["a", "b"],
            ["b", "a"],
            ["a", "a"],
            ["b", "a"],
            ["a", "b"],
        ],
        dtype=object,
    )

    score = mean_average_precision_at_r(y_true, y_pred)

    assert score == pytest.approx((0.5 + 0.25 + 1.0 + 1.0 + 0.0) / 5)


def test_nearest_label_nmi_accepts_ranked_predictions():
    y_true = np.array(["a", "a", "b", "b"], dtype=object)
    y_pred = np.array(
        [
            ["a", "b"],
            ["a", "b"],
            ["b", "a"],
            ["b", "a"],
        ],
        dtype=object,
    )

    assert nearest_label_nmi(y_true, y_pred) == pytest.approx(1.0)
