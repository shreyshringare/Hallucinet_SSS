"""Unit tests for server.calibration.CalibrationTracker (ECE)."""

from server.calibration import CalibrationTracker


def test_empty_tracker_returns_zero_error():
    curve = CalibrationTracker().get_calibration_curve()
    assert curve["calibration_error"] == 0.0
    assert curve["total_samples"] == 0
    assert curve["bins"] == []


def test_perfectly_calibrated_bin():
    t = CalibrationTracker()
    # Bin center 0.55: half correct at confidence in [0.5, 0.6) -> gap ~0.05.
    t.record(0.55, True)
    t.record(0.55, False)
    curve = t.get_calibration_curve()
    assert curve["total_samples"] == 2
    assert len(curve["bins"]) == 1
    assert curve["bins"][0]["actual_accuracy"] == 0.5
    assert curve["calibration_error"] <= 0.06


def test_overconfident_detector_high_ece():
    t = CalibrationTracker()
    # Always 95% confident, always wrong -> ECE near 0.95.
    for _ in range(10):
        t.record(0.95, False)
    curve = t.get_calibration_curve()
    assert curve["calibration_error"] > 0.9
    assert curve["interpretation"] == "Poorly calibrated"


def test_well_calibrated_interpretation():
    t = CalibrationTracker()
    # 90% confidence, 90% accuracy -> tiny gap.
    for i in range(10):
        t.record(0.95, i < 9)
    curve = t.get_calibration_curve()
    assert curve["calibration_error"] < 0.1
    assert curve["interpretation"] == "Well calibrated"


def test_confidence_one_lands_in_top_bin():
    t = CalibrationTracker(n_bins=10)
    t.record(1.0, True)
    curve = t.get_calibration_curve()
    assert curve["bins"][0]["confidence_bin"] == 0.95
