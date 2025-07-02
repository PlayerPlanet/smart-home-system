from pydantic import BaseModel
from typing import Dict, List

class SensorData(BaseModel):
    meta: Dict[str, str]
    results: Dict[str, float]