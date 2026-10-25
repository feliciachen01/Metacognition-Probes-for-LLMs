import numpy as np

from kwik.metrics import accuracy_at_coverage, aurc, auroc, ece, risk_coverage


def test_auroc_perfect_and_reversed():
    y = np.array([0, 0, 1, 1])
    assert auroc(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert auroc(y, np.array([0.9, 0.8, 0.2, 0.1])) == 0.0


def test_ece_calibrated_is_small():
    rng = np.random.default_rng(0)
    r = rng.uniform(size=20000)
    y = (rng.uniform(size=r.size) < r).astype(int)
    assert ece(y, r) < 0.02
    assert ece(y, np.full_like(r, 0.99)) > 0.4


def test_risk_coverage_orders_by_risk():
    y = np.array([1, 0, 0, 1])
    r = np.array([0.9, 0.1, 0.2, 0.8])
    cov, risk = risk_coverage(y, r)
    assert np.allclose(cov, [0.25, 0.5, 0.75, 1.0])
    assert np.allclose(risk, [0, 0, 1 / 3, 0.5])
    assert accuracy_at_coverage(y, r, 0.5) == 1.0
    assert aurc(y, r) < aurc(y, -r)
