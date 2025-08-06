import matplotlib.pyplot as plt
from .models import SensorData, Device
from .rssi_models import base_model
from .heuristic_models import _rank_target
from typing import List, Dict, Optional, Set, Tuple, BinaryIO
from datetime import datetime
import matplotlib.dates as mdates
import numpy as np
from scipy.linalg import eigh
from scipy.optimize import curve_fit, minimize
from skimage import io, color, filters, morphology
from io import BytesIO
from skimage.feature import canny
from datetime import timedelta
import os

def create_plot_from_SensorData(
    all_sensor_data: List[SensorData], 
    selected_keys: Optional[List[str]] = None,
    output_dir: str = "static/img/"
) -> List[str]:
    devices: Dict[str, Dict[str, List[tuple]]] = {}
    for sensor_data in all_sensor_data:
        mac = sensor_data.meta.get("mac", "unknown")
        time_str = sensor_data.meta.get("time")
        t = datetime.strptime(time_str, "%Y/%m/%d:%H:%M:%S")
        for key, value in sensor_data.results.items():
            if selected_keys is None or key in selected_keys:
                devices.setdefault(key, {}).setdefault(mac, []).append((t, value))
    image_paths = []
    for key, mac_dict in devices.items():
        plt.figure(figsize=(8, 4))
        for mac, tv_list in mac_dict.items():
            times, values = zip(*sorted(tv_list))
            plt.plot(times, values, marker="o", label=f"Sensor {mac}")
        plt.title(key)
        plt.xlabel("Time")
        plt.ylabel("rssi")
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.gcf().autofmt_xdate()
        plt.legend()
        plt.tight_layout()
        filename = f"plot_{key}.png"
        plt.savefig(output_dir + filename)
        plt.close()
        image_paths.append(output_dir + filename)
    return image_paths

def update_distance_parameters(data: np.ndarray[float])->str:
    distance_matrix = base_model(data)
    coords = MDS(distance_matrix)
    _save_plot_to_img(coords)
    return "success!"


def _save_plot_to_img(coords: np.ndarray, name: str = "distance_plot"):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(coords[:, 0], coords[:, 1])
    for i in range(len(coords)):
        ax.text(coords[i, 0], coords[i, 1], f'Node {i+1}', fontsize=12)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.savefig("static/img/distance_plot.png")
    plt.close()

def MDS(D, n_components=2):
    # D: distance matrix (n_samples x n_samples)
    n = D.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * H @ (D ** 2) @ H
    eigvals, eigvecs = eigh(B)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx][:n_components]
    eigvecs = eigvecs[:, idx][:, :n_components]
    return eigvecs * np.sqrt(eigvals)

def create_img_mask(image: str | bytes, cleaned = True) -> np.ndarray:
    """create mask using otsu threshold. if cleaned = True, returns morphology-cleaned mask"""
    if type(image) == str:
        image = io.imread(str)
    elif type(image) == bytes: 
        image = io.imread(BytesIO(image))
    else:
        raise TypeError
    if image.shape[-1] == 4:
        image = image[:, :, :3]
    gray = color.rgb2gray(image)
    thresh = filters.threshold_otsu(gray)
    binary_mask = gray > thresh
    if cleaned: 
        cleaned_mask = morphology.binary_closing(binary_mask, morphology.disk(2))
        cleaned = morphology.remove_small_objects(cleaned_mask, min_size=300)
        cleaned = morphology.binary_opening(cleaned_mask, morphology.disk(2))
        return cleaned_mask.astype(np.uint8)
    else:
        return binary_mask.astype(np.uint8)

def create_canny_img_mask(image: str | bytes) -> np.ndarray:
    """create mask using canny edge-detection"""
    if type(image) == str:
        image = io.imread(str)
    elif type(image) == bytes: 
        image = io.imread(BytesIO(image))
    else:
        raise TypeError
    if image.shape[-1] == 4:
        image = image[:, :, :3]
    gray = color.rgb2gray(image)
    edges = canny(gray, sigma=2)
    
    return edges.astype(np.uint8)

def save_heatmap_to_png(data: np.ndarray, name: str):
    plt.imshow(data, cmap='hot')
    plt.title("heatmap")
    plt.axis('off')
    plt.savefig(f"src/model_data/{name}.png")

def fit_calibration_data(sensor_matrix: np.ndarray, position_matrix: np.ndarray)-> Dict[str,float]:
    """Retuns fit parameters N, and rssi_0: sensor_matrix and position_matrix must be same size"""
    if np.shape(sensor_matrix) == np.shape(position_matrix):
        rssi_data, distance_data = extract_valid_data(sensor_matrix, position_matrix)
        # Initial guess
        p0 = [-54.4, 2.305]
        # Bounds for rssi_0 and N
        bounds = ([-100, 0], [-10, 10])  # Lower and upper bounds
        # Fit
        params, covariance = curve_fit(base_model, rssi_data, distance_data, p0=p0, bounds=bounds)
        np.save("src/model_data/params_covariance.npy", covariance)
        np.save("src/model_data/params.npy", params)
        return {"rssi_0":params[0],"N":params[1]}
    else: 
        raise TypeError("sensor_matrix and position_matrix must be same size")
        return None
    
def extract_valid_data(D: np.ndarray, R: np.ndarray):
    n = R.shape[0]
    # Mask out invalid entries
    valid_mask = (R != 0) & (~np.eye(n, dtype=bool))  # exclude diagonal
    # Flatten the indices of all valid entries
    i_idx, j_idx = np.where(valid_mask)
    # Extract corresponding values
    rssi_values = R[i_idx, j_idx]
    distance_values = D[i_idx, j_idx]

    return rssi_values, distance_values

def target_heuristic(
    all_sensor_data: List[SensorData], 
    num_sensors: int,
    selected_keys: Optional[List[str]] = None,
) -> Dict[str, float]:
    devices: Dict[str, Dict[str, List[tuple]]] = {}
    for sensor_data in all_sensor_data:
        mac = sensor_data.meta.get("mac", "unknown")
        time_str = sensor_data.meta.get("time")
        t = datetime.strptime(time_str, "%Y/%m/%d:%H:%M:%S")
        for key, value in sensor_data.results.items():
            if selected_keys is None or key in selected_keys:
                devices.setdefault(key, {}).setdefault(mac, []).append((t, value))
    now = datetime.now()
    S_max = num_sensors
    scored_targets = {
    target_mac: _rank_target(sensor_hits, now, S_max)
    for target_mac, sensor_hits in devices.items()
    }
    return scored_targets

def objective(params, training_data, now, S_max):
    rssi_ref, rssi_min, N_thresh, half_life = params
    mse = 0.0
    for hits_by_sensor, true_score in training_data:
        pred = _rank_target(hits_by_sensor, now, S_max, rssi_ref, rssi_min, int(N_thresh), half_life)
        mse += (pred - true_score) ** 2
        #print(f"Pred: {pred:.3f}, True: {true_score:.3f}, Params: {params}")
    return mse / len(training_data)

def train_heuristic_params(
        all_sensor_data: List[SensorData], 
        S_max: int
):
    """For training to be effective user must rate all devices"""
    now = datetime.now()
    initial_params = [-60.0, -100.0, 6.0, 36000.0]
    training_data = prepare_training_data(all_sensor_data)
    # Optional: parameter bounds
    bounds = [
        (-80, -40),     # rssi_ref
        (-120, -70),    # rssi_min
        (1, 20),        # N_thresh
        (100, 100000)   # half_life
    ]
    res = minimize(objective, initial_params, args=(training_data, now, S_max), method="L-BFGS-B", bounds=bounds)
    return res

import json
from datetime import datetime

# --- Load JSON ratings
def load_user_ratings(path="/model_data/ratings_log.json"):
    device_ratings = {}
    with open(path, "r") as f:
        for line in f:
            entry = json.loads(line)
            device_ratings[entry["device"]] = entry["rating"]
    return device_ratings

# --- Build training tuples
def build_training_data(devices, device_ratings):
    return [
        (sensor_hits, device_ratings[device])
        for device, sensor_hits in devices.items()
        if device in device_ratings
    ]

# --- Combine all
def prepare_training_data(all_sensor_data, selected_keys=None, rating_path="src/model_data/ratings_log.json"):
    devices = {}
    for sensor_data in all_sensor_data:
        mac = sensor_data.meta.get("mac", "unknown")
        time_str = sensor_data.meta.get("time")
        t = datetime.strptime(time_str, "%Y/%m/%d:%H:%M:%S")
        for key, value in sensor_data.results.items():
            if selected_keys is None or key in selected_keys:
                devices.setdefault(key, {}).setdefault(mac, []).append((t, value))
    device_ratings = load_user_ratings(rating_path)
    return build_training_data(devices, device_ratings)

