import os
from unittest.mock import patch

from agentune.runner import _get_mlflow


class TestGetMlflow:
    def test_returns_none_when_no_tracking_uri(self):
        env = os.environ.copy()
        env.pop("MLFLOW_TRACKING_URI", None)
        with patch.dict(os.environ, env, clear=True):
            assert _get_mlflow() is None

    def test_returns_mlflow_when_tracking_uri_set(self):
        with patch.dict(os.environ, {"MLFLOW_TRACKING_URI": "http://localhost:5001"}):
            ml = _get_mlflow()
            assert ml is not None

    def test_sets_tracking_uri(self):
        with patch.dict(os.environ, {"MLFLOW_TRACKING_URI": "http://localhost:5001"}):
            with patch("agentune.runner._mlflow") as mock_ml:
                mock_ml.__bool__ = lambda self: True
                _get_mlflow()
                mock_ml.set_tracking_uri.assert_called_once_with("http://localhost:5001")
