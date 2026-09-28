import logging
from typing import List, Dict, Any, Optional

from backend.inference.uci_forecaster import UCIForecaster

logger = logging.getLogger("gridbalance.service.forecasting")

_uci_forecaster: Optional[UCIForecaster] = None

def get_uci_forecaster() -> UCIForecaster:
    global _uci_forecaster
    if _uci_forecaster is None:
        _uci_forecaster = UCIForecaster()
    return _uci_forecaster

class ForecastingService:
    def __init__(self):
        self.forecaster = get_uci_forecaster()

    def predict_next_hour(self, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        return self.forecaster.predict_next_hour(history)
