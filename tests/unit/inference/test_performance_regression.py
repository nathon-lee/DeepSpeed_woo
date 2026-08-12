# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

import pytest

from unit.inference.test_inference import _assert_performance_regression


def test_performance_regression_check_is_opt_in(monkeypatch):
    monkeypatch.delenv("DS_INFERENCE_PERF_MAX_SLOWDOWN", raising=False)
    _assert_performance_regression(10.0, 100.0)


def test_performance_regression_check_rejects_slowdown(monkeypatch):
    monkeypatch.setenv("DS_INFERENCE_PERF_MAX_SLOWDOWN", "1.10")
    with pytest.raises(AssertionError, match="performance regression"):
        _assert_performance_regression(10.0, 11.1)


@pytest.mark.parametrize("value", ["0", "-1", "not-a-number"])
def test_performance_regression_check_validates_limit(monkeypatch, value):
    monkeypatch.setenv("DS_INFERENCE_PERF_MAX_SLOWDOWN", value)
    with pytest.raises(ValueError, match="positive number"):
        _assert_performance_regression(10.0, 10.0)
