"""
Smart Home IoT System - Heuristic Device Ranking Models

This module implements intelligent algorithms for scoring and ranking IoT devices
based on their presence patterns, signal quality, and behavioral characteristics.
The heuristic models analyze multi-sensor RSSI data to predict device importance
and relevance in smart home environments.

Key Features:
- Multi-factor device importance scoring algorithms
- Temporal pattern analysis with exponential decay
- Signal quality assessment and normalization
- Sensor coverage and detection frequency analysis
- Tunable parameters for different environments and use cases

The ranking algorithms consider multiple factors to provide comprehensive
device importance scores that help users identify the most relevant IoT
devices in their environment.

Scoring Factors:
1. Signal Quality: RSSI strength normalized to expected range
2. Detection Frequency: Number of sensor hits with soft thresholding  
3. Sensor Coverage: Proportion of sensors that detect the device
4. Temporal Decay: Recent activity weighted higher than historical data

Author: Smart Home IoT Team
Version: 1.0.0
"""

import numpy as np
from datetime import timedelta

def _rank_targetv0(hits_by_sensor, now, S_max, rssi_ref=-60, rssi_min=-100, N_thresh=6, half_life=36000):
    """
    Original device ranking algorithm (legacy version).
    
    Simple multiplicative scoring model that combines detection count, sensor
    coverage, temporal decay, and signal quality into a single importance score.
    
    Args:
        hits_by_sensor (Dict): Device detections by sensor {sensor_id: [(time, rssi), ...]}
        now (datetime): Current timestamp for temporal decay calculation
        S_max (int): Maximum number of sensors in system for normalization
        rssi_ref (float): Reference RSSI for good signal quality (default: -60 dBm)
        rssi_min (float): Minimum detectable RSSI threshold (default: -100 dBm)
        N_thresh (int): Hit count threshold for full scoring (default: 6)
        half_life (float): Temporal decay half-life in seconds (default: 36000 = 10 hours)
        
    Returns:
        float: Device importance score [0.0, 1.0]
        
    Scoring Components:
        - C_hits: Hit count factor with hard threshold
        - C_sensors: Sensor coverage factor with power scaling
        - C_decay: Exponential temporal decay from last detection
        - C_rssi: Average RSSI normalized to [0, 1] range
        
    Final Score: C_hits × C_sensors × C_decay × C_rssi
    
    Example:
        ```python
        hits = {
            "sensor_1": [(datetime(2025,1,15,10,0), -65), (datetime(2025,1,15,11,0), -62)],
            "sensor_2": [(datetime(2025,1,15,10,30), -70)]
        }
        score = _rank_targetv0(hits, datetime.now(), S_max=3)
        ```
        
    Note:
        Legacy function maintained for compatibility. Consider using _rank_target()
        for improved temporal weighting and more sophisticated scoring logic.
    """
    all_hits = [hit for hits in hits_by_sensor.values() for hit in hits]
    if not all_hits:
        return 0.0

    N = len(all_hits)
    S_t = len(hits_by_sensor)
    last_hit_time = max(hit[0] for hit in all_hits)
    T_last: timedelta = now - last_hit_time
    T_last = T_last.total_seconds()
    RSSI_avg = np.mean([hit[1] for hit in all_hits])

    C_hits = min(1.0, N / N_thresh)
    C_sensors = (S_t / S_max) ** 1.5
    decay_lambda = np.log(2) / half_life
    C_decay = np.exp(-decay_lambda * T_last)
    C_rssi = min(1.0, max(0.0, (RSSI_avg - rssi_min) / (rssi_ref - rssi_min)))

    score = C_hits * C_sensors * C_decay * C_rssi
    return round(score, 3)

def _rank_target(
    hits_by_sensor,
    now,
    S_max,
    rssi_ref=-60,
    rssi_min=-100,
    N_thresh=6,
    half_life=36000
):
    """
    Advanced device ranking algorithm with temporal weighting.
    
    Sophisticated scoring model that uses time-weighted RSSI averaging and
    soft thresholding for more accurate device importance assessment. Improves
    upon the original algorithm with better temporal handling and numerical stability.
    
    Args:
        hits_by_sensor (Dict): Device detections {sensor_id: [(timestamp, rssi), ...]}
        now (datetime): Current timestamp for decay calculations
        S_max (int): Maximum sensors for coverage normalization
        rssi_ref (float): Reference RSSI for excellent signal (default: -60 dBm)
        rssi_min (float): Minimum meaningful RSSI (default: -100 dBm)  
        N_thresh (int): Hit count soft threshold (default: 6)
        half_life (float): Temporal decay half-life in seconds (default: 36000)
        
    Returns:
        float: Device importance score [0.0, 1.0], clipped and normalized
        
    Advanced Features:
        1. Time-Weighted RSSI: Recent measurements weighted exponentially higher
        2. Soft Thresholding: Smooth hit count bonuses using exponential functions
        3. Normalized Sensor Coverage: Linear scaling by maximum available sensors
        4. Robust Score Combination: Multiplicative model with overflow protection
        
    Mathematical Model:
        ```
        time_weight(t) = exp(-(now - t) / half_life)
        weighted_rssi = Σ(rssi_weight × time_weight) / Σ(time_weight)
        hit_bonus = 1 - exp(-hit_count / N_thresh)
        sensor_bonus = active_sensors / S_max
        final_score = weighted_rssi × hit_bonus × sensor_bonus
        ```
        
    Example:
        ```python
        # Complex detection pattern
        hits = {
            "sensor_1": [
                (datetime(2025,1,15,14,0), -60),   # Recent, strong
                (datetime(2025,1,15,10,0), -65)    # Older, weaker
            ],
            "sensor_2": [(datetime(2025,1,15,13,0), -62)],
            "sensor_3": [(datetime(2025,1,15,12,0), -68)]
        }
        
        score = _rank_target(hits, datetime.now(), S_max=5)
        # Returns weighted score considering time decay and multi-sensor coverage
        ```
        
    Improvements over v0:
        - Time-weighted averaging instead of simple mean
        - Soft thresholding eliminates hard cutoffs
        - Better numerical stability and score normalization
        - More intuitive parameter behavior
        
    Note:
        Recommended for production use. Parameters can be optimized using
        machine learning on user rating data via train_heuristic_params().
    """
    all_hits = [hit for hits in hits_by_sensor.values() for hit in hits]
    if len(all_hits) == 0:
        return 0.0

    # Time-decay weights
    def time_decay(t):
        dt = (now - t).total_seconds()
        return np.exp(-dt / half_life)

    weighted_rssi = 0.0
    total_weight = 0.0
    for sensor, hits in hits_by_sensor.items():
        for t, rssi in hits:
            rssi_weight = (rssi - rssi_min) / (rssi_ref - rssi_min)
            decay_weight = time_decay(t)
            total_weight += decay_weight
            weighted_rssi += decay_weight * rssi_weight

    if total_weight == 0:
        return 0.0

    avg_rssi_score = weighted_rssi / total_weight  # ∈ [0, 1] roughly

    # Soft thresholds
    hit_bonus = 1 - np.exp(-len(all_hits) / N_thresh)
    sensor_bonus = len(hits_by_sensor) / S_max  # normalize to max

    # Final score combines: signal strength, #hits, #sensors
    score = avg_rssi_score * hit_bonus * sensor_bonus

    # Normalize and clip
    return min(max(score, 0.0), 1.0)
