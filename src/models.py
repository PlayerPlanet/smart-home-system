"""
Smart Home IoT System - Data Models

This module defines the core Pydantic data models used throughout the smart home
IoT system for data validation, serialization, and API communication.

The models provide type-safe data structures for:
- Sensor RSSI measurements and metadata
- IoT device identification and tracking  
- User device importance ratings

All models use Pydantic for automatic validation, serialization to JSON,
and integration with FastAPI endpoints.

Author: Smart Home IoT Team
Version: 1.0.0
"""

from pydantic import BaseModel
from typing import Dict, List, Optional

class SensorData(BaseModel):
    """
    RSSI sensor measurement data with metadata.
    
    Represents a single sensor's RSSI measurements of multiple IoT devices
    at a specific point in time, along with sensor identification and 
    timing information.
    
    Attributes:
        meta (Dict[str, str]): Sensor metadata including MAC address, timestamp,
            and other identifying information. Common keys:
            - "mac": Sensor MAC address (required)
            - "time": Timestamp in "YYYY/MM/DD:HH:MM:SS" format
            - "location": Optional sensor location description
            
        results (Dict[str, float]): RSSI measurements by device MAC address.
            Keys are device MAC addresses, values are RSSI measurements in dBm.
            Example: {"device_001": -65.2, "device_002": -72.8}
    
    Example:
        ```python
        sensor_data = SensorData(
            meta={"mac": "sensor_001", "time": "2025/01/15:14:30:00"},
            results={"device_001": -65.2, "device_002": -72.8}
        )
        ```
    
    Note:
        RSSI values are typically negative dBm values, where higher values
        (closer to 0) indicate stronger signals and closer proximity.
    """
    meta: Dict[str, str]
    results: Dict[str, float]

class Device(BaseModel):
    """
    IoT device identification model.
    
    Represents a unique IoT device in the smart home system, identified
    primarily by MAC address with optional human-readable naming.
    
    Attributes:
        name (Optional[str]): Human-readable device name or description.
            Defaults to None if not provided. Used for display purposes.
            
        mac (str): Unique MAC address identifier (required). Must be a valid
            MAC address string format. Used as primary device identifier
            throughout the system.
    
    Example:
        ```python
        # Device with name
        device = Device(name="Living Room Speaker", mac="aa:bb:cc:dd:ee:ff")
        
        # Device with MAC only
        device = Device(mac="11:22:33:44:55:66")
        ```
    
    Configuration:
        frozen=True: Makes instances immutable after creation, ensuring
        device identity remains constant throughout the system lifecycle.
    
    Note:
        MAC addresses serve as unique identifiers and should follow standard
        format (e.g., "aa:bb:cc:dd:ee:ff"). The immutable nature ensures
        devices can be safely used as dictionary keys and in sets.
    """
    name: Optional[str] = None
    mac: str

    class Config:
        frozen = True

class Rating(BaseModel):
    """
    User rating for device importance.
    
    Captures user-provided importance ratings for IoT devices, used to train
    and improve the automatic device ranking algorithms through machine learning.
    
    Attributes:
        device (str): MAC address of the device being rated. Must match
            a device MAC address present in the sensor data.
            
        rating (float): Importance rating score from 0.0 to 1.0, where:
            - 0.0: Not important/never present
            - 0.5: Moderately important/sometimes present  
            - 1.0: Very important/always present
    
    Example:
        ```python
        # High importance device (always present)
        rating = Rating(device="aa:bb:cc:dd:ee:ff", rating=0.95)
        
        # Low importance device (rarely seen)
        rating = Rating(device="11:22:33:44:55:66", rating=0.15)
        ```
    
    Usage:
        Ratings are collected through the web interface and stored in JSON
        format for training heuristic models. The system uses these ratings
        to learn user preferences and improve automatic device scoring.
    
    Note:
        Rating values should be between 0.0 and 1.0. Values outside this
        range may cause issues with machine learning algorithms.
    """
    device: str
    rating: float