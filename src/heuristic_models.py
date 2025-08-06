import numpy as np
from datetime import timedelta

def _rank_targetv0(hits_by_sensor, now, S_max, rssi_ref=-60, rssi_min=-100, N_thresh=6, half_life=36000):
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
