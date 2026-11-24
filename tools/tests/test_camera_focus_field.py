import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from camera_focus_field import direction, local, predict, unit


def grid():
    return np.array([(x, y) for x in range(8, 123, 16) for y in range(8, 123, 16)])


def test_training_recovers_unoriented_normal_across_reversed_slopes():
    xy = grid()
    axis = unit(82)
    frames = [
        np.column_stack([xy, intercept + sign * 0.01 * (xy @ axis)])
        for intercept, sign in [(2, -1), (2.3, 1)]
    ]
    angle, individual = direction(frames)
    assert angle == 82
    assert np.all(individual == 82)


def test_withheld_sigma_never_enters_its_prediction():
    xy = grid()
    frame = np.column_stack([xy, 1.5 + 0.004 * xy[:, 1]])
    size = np.array([128, 128])

    def first():
        return next(v for v in predict(frame, unit(90), size, ablate=True) if v[0] == 0)

    before = first()
    frame[0, 2] += 100
    after = first()
    for a, b in zip(before[:-1], after[:-1]):
        np.testing.assert_allclose(a, b)
    assert before[-1] == after[-1]
    np.testing.assert_allclose(before[2], [0, 0.004], atol=1e-12)


def test_collinear_neighbor_support_is_flagged():
    frame = np.array([[i, i, 0.1 * i] for i in range(6)])
    _, gradient, valid = local(frame, np.array([1, 1]))
    assert not valid
    assert np.all(gradient == 0)
