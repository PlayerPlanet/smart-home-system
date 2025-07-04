from pydantic import BaseModel
from typing import Dict, List, Optional

class SensorData(BaseModel):
    meta: Dict[str, str]
    results: Dict[str, float]

class Device(BaseModel):
    name: Optional[str] = None
    mac: str

    class Config:
        frozen = True