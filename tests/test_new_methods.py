"""Unit tests for the two new fairness methods:
  - calibrated_equalized_odds  (Hardt et al., NeurIPS 2016)
  - reject_option              (Kamiran et al., ICDM 2012)
"""

from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers shared across tests
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)


def _make_binary_dataset(n: int = 400, base_rate_diff: float = 0.2, seed: int = 0):
    """Synthetic binary classification dataset with a controlled base-rate difference.

    Group A=0: P(Y=1) = 0.5
    Group A=1: P(Y=1) = 0.5 + base_rate_diff

    ``base_rate_diff`` is the difference in *label* rates between groups (ground
    truth), not the DP gap of predictions.

    Returns y_true, y_proba (as noisy version of y_true), A.
    """
    rng = np.random.default_rng(seed)
    A = rng.integers(0, 2, size=n)

    p_pos = np.where(A == 0, 0.5, 0.5 + base_rate_diff)
    y_true = (rng.random(n) < p_pos).astype(int)

    # Calibrated-ish scores: add noise to the true label probability
    y_proba = np.clip(p_pos + rng.normal(0, 0.15, size=n), 0.01, 0.99)
    return y_true, y_proba, A


GRID = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


# ---------------------------------------------------------------------------
# calibrated_equalized_odds helpers
# ---------------------------------------------------------------------------

class TestRocPointsForGroup:
    def test_shape(self):
        from src.benchmark.methods import _roc_points_for_group

        y = np.array([0, 1, 0, 1, 1])
        p = np.array([0.2, 0.8, 0.3, 0.7, 0.6])
        grid = [0.3, 0.5, 0.7]
        result = _roc_points_for_group(y, p, grid)

        assert "thresholds" in result
        assert "tpr" in result
        assert "fpr" in result
        assert "error" in result
        assert len(result["tpr"]) == len(grid)
        assert len(result["fpr"]) == len(grid)

    def test_tpr_fpr_range(self):
        from src.benchmark.methods import _roc_points_for_group

        y_true, y_proba, A = _make_binary_dataset()
        mask = A == 0
        result = _roc_points_for_group(y_true[mask], y_proba[mask], GRID)

        assert np.all(result["tpr"] >= 0) and np.all(result["tpr"] <= 1)
        assert np.all(result["fpr"] >= 0) and np.all(result["fpr"] <= 1)

    def test_tpr_decreases_with_threshold(self):
        """Higher threshold → stricter → lower or equal TPR (non-increasing within float tolerance)."""
        from src.benchmark.methods import _roc_points_for_group

        y_true, y_proba, A = _make_binary_dataset(n=1000)
        mask = A == 0
        result = _roc_points_for_group(y_true[mask], y_proba[mask], sorted(GRID))

        # TPR should be non-increasing as threshold increases (allow ε for float precision)
        assert np.all(np.diff(result["tpr"]) <= 1e-9), (
            "TPR should be non-increasing with threshold (within float precision)"
        )

    def test_all_positive_predictions_at_low_threshold(self):
        """Very low threshold → all predictions positive → TPR=1, FPR=1."""
        from src.benchmark.methods import _roc_points_for_group

        y = np.array([0, 1, 0, 1])
        p = np.array([0.9, 0.9, 0.9, 0.9])
        result = _roc_points_for_group(y, p, [0.01])
        assert result["tpr"][0] == pytest.approx(1.0)
        assert result["fpr"][0] == pytest.approx(1.0)


class TestChooseCalibratedEqualizedOdds:
    def test_returns_correct_structure(self):
        from src.benchmark.methods import choose_calibrated_equalized_odds

        y_true, y_proba, A = _make_binary_dataset()
        result = choose_calibrated_equalized_odds(y_true, y_proba, A, GRID, seed=7)

        assert 0 in result and 1 in result
        assert "seed" in result
        assert "weights" in result[0] and "thresholds" in result[0]
        assert "weights" in result[1] and "thresholds" in result[1]

    def test_mixing_weights_sum_to_one(self):
        from src.benchmark.methods import choose_calibrated_equalized_odds

        y_true, y_proba, A = _make_binary_dataset()
        result = choose_calibrated_equalized_odds(y_true, y_proba, A, GRID, seed=0)

        assert result[0]["weights"].sum() == pytest.approx(1.0, abs=1e-6)
        assert result[1]["weights"].sum() == pytest.approx(1.0, abs=1e-6)

    def test_mixing_weights_non_negative(self):
        from src.benchmark.methods import choose_calibrated_equalized_odds

        y_true, y_proba, A = _make_binary_dataset()
        result = choose_calibrated_equalized_odds(y_true, y_proba, A, GRID, seed=0)

        assert np.all(result[0]["weights"] >= -1e-9)
        assert np.all(result[1]["weights"] >= -1e-9)

    def test_thresholds_aligned_with_grid(self):
        from src.benchmark.methods import choose_calibrated_equalized_odds

        y_true, y_proba, A = _make_binary_dataset()
        result = choose_calibrated_equalized_odds(y_true, y_proba, A, GRID, seed=0)

        assert len(result[0]["thresholds"]) == len(GRID)
        assert len(result[1]["thresholds"]) == len(GRID)

    def test_single_group_fallback(self):
        """When only one group is present, should return without crashing."""
        from src.benchmark.methods import choose_calibrated_equalized_odds

        n = 50
        y_true = np.random.randint(0, 2, n)
        y_proba = np.random.rand(n)
        A = np.zeros(n, dtype=int)  # only group 0

        result = choose_calibrated_equalized_odds(y_true, y_proba, A, GRID, seed=0)
        assert 0 in result and 1 in result

    def test_seed_preserved_in_output(self):
        from src.benchmark.methods import choose_calibrated_equalized_odds

        y_true, y_proba, A = _make_binary_dataset()
        result = choose_calibrated_equalized_odds(y_true, y_proba, A, GRID, seed=123)
        assert result["seed"] == 123


class TestApplyCalibratedEqualizedOdds:
    def test_output_binary(self):
        from src.benchmark.methods import choose_calibrated_equalized_odds, apply_calibrated_equalized_odds

        y_true, y_proba, A = _make_binary_dataset()
        mixing = choose_calibrated_equalized_odds(y_true, y_proba, A, GRID, seed=0)
        y_pred = apply_calibrated_equalized_odds(y_proba, A, mixing)

        assert set(np.unique(y_pred)).issubset({0, 1})

    def test_output_length(self):
        from src.benchmark.methods import choose_calibrated_equalized_odds, apply_calibrated_equalized_odds

        y_true, y_proba, A = _make_binary_dataset(n=200)
        mixing = choose_calibrated_equalized_odds(y_true, y_proba, A, GRID, seed=0)
        y_pred = apply_calibrated_equalized_odds(y_proba, A, mixing)

        assert len(y_pred) == 200

    def test_deterministic_with_same_seed(self):
        from src.benchmark.methods import choose_calibrated_equalized_odds, apply_calibrated_equalized_odds

        y_true, y_proba, A = _make_binary_dataset()
        mixing = choose_calibrated_equalized_odds(y_true, y_proba, A, GRID, seed=42)
        y1 = apply_calibrated_equalized_odds(y_proba, A, mixing)
        y2 = apply_calibrated_equalized_odds(y_proba, A, mixing)

        np.testing.assert_array_equal(y1, y2)

    def test_equalized_odds_gap_reduced_vs_baseline(self):
        """Calibrated EO should reduce the equalized-odds gap compared to standard 0.5 threshold.

        Tolerance of +0.05 accounts for statistical noise at n=2000 and the
        inherent variance of the randomized mixing strategy over a small grid.
        The LP guarantees the *expected* gap is ≤ baseline; the realized gap
        may differ slightly due to the finite-sample randomization.
        """
        from src.benchmark.methods import choose_calibrated_equalized_odds, apply_calibrated_equalized_odds
        from src.fairness import equalized_odds_gap

        # Use larger, more controlled dataset for statistical power
        y_true, y_proba, A = _make_binary_dataset(n=2000, base_rate_diff=0.3, seed=1)
        mixing = choose_calibrated_equalized_odds(y_true, y_proba, A, GRID, seed=0)
        y_pred_ceo = apply_calibrated_equalized_odds(y_proba, A, mixing)

        y_pred_base = (y_proba >= 0.5).astype(int)

        gap_ceo = equalized_odds_gap(y_true, y_pred_ceo, A)
        gap_base = equalized_odds_gap(y_true, y_pred_base, A)

        # Calibrated EO should achieve a smaller or equal gap
        # Tolerance accounts for finite-sample randomization variance
        assert gap_ceo <= gap_base + 0.05, (
            f"Calibrated EO gap ({gap_ceo:.4f}) should be ≤ baseline gap ({gap_base:.4f})"
        )

    def test_handles_single_group_at_prediction(self):
        """No crash when A contains only one group value at prediction time."""
        from src.benchmark.methods import choose_calibrated_equalized_odds, apply_calibrated_equalized_odds

        y_true, y_proba, A = _make_binary_dataset(n=100)
        mixing = choose_calibrated_equalized_odds(y_true, y_proba, A, GRID, seed=0)

        A_single = np.zeros(50, dtype=int)  # all group 0
        y_pred = apply_calibrated_equalized_odds(y_proba[:50], A_single, mixing)
        assert len(y_pred) == 50
        assert set(np.unique(y_pred)).issubset({0, 1})


# ---------------------------------------------------------------------------
# reject_option helpers
# ---------------------------------------------------------------------------

class TestApplyRejectOption:
    def test_output_binary(self):
        from src.benchmark.methods import _apply_reject_option

        y_true, y_proba, A = _make_binary_dataset()
        y_pred = _apply_reject_option(y_proba, A, margin=0.1, disadvantaged_group=1)
        assert set(np.unique(y_pred)).issubset({0, 1})

    def test_output_length(self):
        from src.benchmark.methods import _apply_reject_option

        n = 150
        y_proba = np.random.rand(n)
        A = np.random.randint(0, 2, n)
        y_pred = _apply_reject_option(y_proba, A, margin=0.15, disadvantaged_group=0)
        assert len(y_pred) == n

    def test_zero_margin_equals_baseline(self):
        """With margin=0 the reject option rule is equivalent to the standard threshold."""
        from src.benchmark.methods import _apply_reject_option

        y_true, y_proba, A = _make_binary_dataset()
        y_base = (y_proba >= 0.5).astype(int)
        y_ro = _apply_reject_option(y_proba, A, margin=0.0, disadvantaged_group=1)
        np.testing.assert_array_equal(y_base, y_ro)

    def test_max_margin_pushes_boundary_predictions(self):
        """With margin=0.5 every sample is in the critical region."""
        from src.benchmark.methods import _apply_reject_option

        n = 200
        y_proba = np.full(n, 0.5)  # all scores exactly at boundary
        A = np.array([0] * 100 + [1] * 100)
        y_pred = _apply_reject_option(y_proba, A, margin=0.49, disadvantaged_group=1)

        # Disadvantaged (A=1): all in margin → predict 1
        assert np.all(y_pred[A == 1] == 1)
        # Privileged (A=0): all in margin → predict 0
        assert np.all(y_pred[A == 0] == 0)

    def test_outside_margin_unaffected(self):
        """Predictions outside the critical region should follow the 0.5 threshold."""
        from src.benchmark.methods import _apply_reject_option

        # Score 0.9 is well outside margin=0.1 → standard prediction
        y_proba = np.array([0.9, 0.1, 0.5])
        A = np.array([0, 1, 0])
        y_pred = _apply_reject_option(y_proba, A, margin=0.1, disadvantaged_group=1)

        assert y_pred[0] == 1   # 0.9 ≥ 0.5, A=0 (not in margin) → 1
        assert y_pred[1] == 0   # 0.1 < 0.5, A=1 (not in margin) → 0


class TestChooseRejectOptionMargin:
    def test_returns_required_keys(self):
        from src.benchmark.methods import choose_reject_option_margin

        y_true, y_proba, A = _make_binary_dataset()
        result = choose_reject_option_margin(y_true, y_proba, A)

        assert "margin" in result
        assert "disadvantaged_group" in result

    def test_margin_in_valid_range(self):
        from src.benchmark.methods import choose_reject_option_margin

        y_true, y_proba, A = _make_binary_dataset()
        result = choose_reject_option_margin(y_true, y_proba, A)

        assert 0.0 <= result["margin"] <= 0.5

    def test_disadvantaged_group_is_binary(self):
        from src.benchmark.methods import choose_reject_option_margin

        y_true, y_proba, A = _make_binary_dataset()
        result = choose_reject_option_margin(y_true, y_proba, A)

        assert result["disadvantaged_group"] in {0, 1}

    def test_dp_gap_reduced_vs_baseline(self):
        """Reject option should reduce the DP gap compared to the standard 0.5 threshold.

        The margin is tuned on the *same* validation dataset (here we pass the
        same data as val and test for simplicity), so the DP gap should be at
        most equal to the baseline.  A small tolerance of 0.005 accounts for
        floating-point rounding when the best margin produces a gap that is
        essentially identical to baseline at machine precision.
        """
        from src.benchmark.methods import choose_reject_option_margin, _apply_reject_option
        from src.fairness import demographic_parity

        y_true, y_proba, A = _make_binary_dataset(n=2000, base_rate_diff=0.3, seed=2)
        info = choose_reject_option_margin(y_true, y_proba, A)
        y_pred_ro = _apply_reject_option(
            y_proba, A, info["margin"], info["disadvantaged_group"]
        )
        y_pred_base = (y_proba >= 0.5).astype(int)

        gap_ro = abs(float(demographic_parity(y_pred_ro, A)))
        gap_base = abs(float(demographic_parity(y_pred_base, A)))

        assert gap_ro <= gap_base + 0.005, (
            f"Reject option DP gap ({gap_ro:.4f}) should be ≤ baseline ({gap_base:.4f})"
        )

    def test_single_group_no_crash(self):
        """Should not crash when only one group is present."""
        from src.benchmark.methods import choose_reject_option_margin

        n = 50
        y_true = np.random.randint(0, 2, n)
        y_proba = np.random.rand(n)
        A = np.ones(n, dtype=int)  # only group 1

        result = choose_reject_option_margin(y_true, y_proba, A)
        assert "margin" in result


# ---------------------------------------------------------------------------
# Integration: runner._train_method_on_data + _predict_with_method
# ---------------------------------------------------------------------------

class TestRunnerIntegration:
    """Quick smoke-test that both new methods train + predict without errors."""

    @pytest.fixture
    def data(self):
        rng = np.random.default_rng(0)
        n = 300
        X_train = rng.standard_normal((n, 5))
        y_train = (X_train[:, 0] > 0).astype(int)
        A_train = (X_train[:, 1] > 0).astype(int)
        X_val = rng.standard_normal((80, 5))
        y_val = (X_val[:, 0] > 0).astype(int)
        A_val = (X_val[:, 1] > 0).astype(int)
        X_test = rng.standard_normal((100, 5))
        A_test = (X_test[:, 1] > 0).astype(int)
        return X_train, y_train, A_train, X_val, y_val, A_val, X_test, A_test

    @pytest.mark.parametrize("method", ["calibrated_equalized_odds", "reject_option"])
    def test_train_and_predict(self, data, method):
        from src.benchmark.runner import _train_method_on_data, _predict_with_method

        X_train, y_train, A_train, X_val, y_val, A_val, X_test, A_test = data
        grid = [0.3, 0.4, 0.5, 0.6, 0.7]
        config = {}

        model, thresholds = _train_method_on_data(
            method, X_train, y_train, A_train, X_val, y_val, A_val,
            seed=0, threshold_grid=grid, config=config,
        )
        assert model is not None
        assert thresholds is not None

        y_pred, y_proba = _predict_with_method(method, model, thresholds, X_test, A_test)
        assert len(y_pred) == 100
        assert set(np.unique(y_pred)).issubset({0, 1})
        assert len(y_proba) == 100
        assert np.all(y_proba >= 0) and np.all(y_proba <= 1)

    @pytest.mark.parametrize("method", ["calibrated_equalized_odds", "reject_option"])
    def test_metrics_computable(self, data, method):
        """Metrics must be computable (no NaN/crash) after prediction."""
        from src.benchmark.runner import _train_method_on_data, _predict_with_method
        from src.benchmark.metrics import compute_metrics

        X_train, y_train, A_train, X_val, y_val, A_val, X_test, A_test = data
        y_test = (X_test[:, 0] > 0).astype(int)
        grid = [0.3, 0.4, 0.5, 0.6, 0.7]

        model, thresholds = _train_method_on_data(
            method, X_train, y_train, A_train, X_val, y_val, A_val,
            seed=5, threshold_grid=grid, config={},
        )
        y_pred, y_proba = _predict_with_method(method, model, thresholds, X_test, A_test)
        metrics = compute_metrics(y_test, y_pred, y_proba, A_test)

        assert "auc" in metrics
        assert "dp_gap" in metrics
        assert "eo_gap" in metrics
        # dp_gap and eo_gap should be finite (may be NaN only if single group)
        assert np.isfinite(metrics["accuracy"])
