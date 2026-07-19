"""Forecasting models subpackage — lazy imports to avoid hard dependency failures."""

from src.models.base import BaseForecaster

def _try_import(module_path, class_name):
    try:
        import importlib
        mod = importlib.import_module(module_path)
        return getattr(mod, class_name)
    except (ImportError, ModuleNotFoundError):
        return None

SARIMAForecaster = _try_import("src.models.sarima_model", "SARIMAForecaster")
ProphetForecaster = _try_import("src.models.prophet_model", "ProphetForecaster")
LightGBMForecaster = _try_import("src.models.lightgbm_model", "LightGBMForecaster")
TFTForecaster = _try_import("src.models.tft_model", "TFTForecaster")
ChronosForecaster = _try_import("src.models.chronos_model", "ChronosForecaster")

__all__ = [
    "BaseForecaster",
    "SARIMAForecaster",
    "ProphetForecaster",
    "LightGBMForecaster",
    "TFTForecaster",
    "ChronosForecaster",
]
