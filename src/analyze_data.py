"""
Smart Home IoT System - Data Analysis and Visualization

This module provides comprehensive data analysis, visualization, and machine
learning capabilities for RSSI sensor data processing. It handles time-series
analysis, device ranking, calibration optimization, and floor plan processing.

Key Capabilities:
- Time-series plotting and visualization of RSSI data
- Multidimensional scaling (MDS) for sensor position estimation  
- Image processing for floor plan mask generation
- Machine learning model training and parameter optimization
- Device importance ranking using heuristic algorithms
- Calibration data processing and parameter fitting

The module integrates mathematical optimization techniques with practical
IoT data processing to provide actionable insights for smart home systems.

Components:
- Visualization: matplotlib-based plotting for time-series and spatial data
- Calibration: scipy optimization for distance model parameter fitting
- Image Processing: scikit-image for floor plan analysis and mask generation
- Machine Learning: training algorithms for device importance prediction

Author: Smart Home IoT Team
Version: 1.0.0
"""

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
    """
    Generate time-series plots for device RSSI measurements.
    
    Creates individual matplotlib plots showing RSSI signal strength over time
    for selected devices, with separate traces for each sensor. Useful for
    visualizing device presence patterns, signal quality, and temporal behavior.
    
    Args:
        all_sensor_data (List[SensorData]): Complete dataset of sensor measurements
        selected_keys (Optional[List[str]]): Device MAC addresses to plot. 
            If None, plots all devices found in data.
        output_dir (str): Directory for saving plot image files (default: "static/img/")
        
    Returns:
        List[str]: File paths of generated plot images
        
    Generated Plots:
        - X-axis: Time (formatted as HH:MM)
        - Y-axis: RSSI values in dBm
        - Multiple lines: One per sensor detecting the device
        - Legend: Sensor identifiers
        - Title: Device MAC address
        
    Example:
        ```python
        # Plot specific devices
        plots = create_plot_from_SensorData(
            sensor_data, 
            selected_keys=["device_001", "device_002"]
        )
        # Returns: ["static/img/plot_device_001.png", "static/img/plot_device_002.png"]
        ```
        
    Note:
        Plot files are named "plot_{device_mac}.png" and saved to output_dir.
        Time axis automatically formats for readability with date rotation.
    """
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
    """
    Update and optimize distance model parameters from calibration data.
    
    Processes RSSI calibration matrix to estimate optimal distance relationships
    between sensors using multidimensional scaling (MDS) and saves visualization.
    
    Args:
        data (np.ndarray[float]): RSSI measurements matrix between sensor pairs
        
    Returns:
        str: "success!" confirmation message
        
    Process:
        1. Convert RSSI matrix to distance matrix using base_model()
        2. Apply MDS to estimate 2D sensor coordinates  
        3. Generate and save scatter plot of estimated positions
        
    Example:
        ```python
        # RSSI matrix from calibration
        rssi_matrix = np.array([
            [0, -65, -70],
            [-65, 0, -68], 
            [-70, -68, 0]
        ])
        result = update_distance_parameters(rssi_matrix)
        ```
        
    Note:
        Saves visualization as "distance_plot.png" showing estimated sensor layout.
    """
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
    """
    Multidimensional Scaling for dimensionality reduction and visualization.
    
    Classical MDS algorithm that embeds high-dimensional distance relationships
    into lower-dimensional Euclidean space while preserving distance structure.
    Used for sensor position estimation from RSSI distance measurements.
    
    Args:
        D (np.ndarray): Distance matrix (n_samples x n_samples)
        n_components (int): Target dimensionality (default: 2 for 2D visualization)
        
    Returns:
        np.ndarray: Embedded coordinates (n_samples x n_components)
        
    Mathematical Algorithm:
        1. Double center the squared distance matrix: B = -0.5 * H * D² * H
        2. Eigendecompose B to get eigenvalues λᵢ and eigenvectors vᵢ  
        3. Select top k eigenvalues and eigenvectors
        4. Coordinates: X = V_k * √Λ_k
        
    Where H = I - (1/n) * 1 * 1ᵀ is the centering matrix.
    
    Example:
        ```python
        # Distance matrix between 4 points
        distances = np.array([
            [0, 1, 2, 3],
            [1, 0, 1, 2], 
            [2, 1, 0, 1],
            [3, 2, 1, 0]
        ])
        coords = MDS(distances)  # Returns 4x2 coordinate array
        ```
        
    Note:
        Assumes Euclidean distance relationships. Works best when input
        distances satisfy triangle inequality and metric properties.
    """
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
    """
    Generate binary mask from floor plan image using Otsu thresholding.
    
    Processes floor plan images to create binary masks distinguishing walls
    from free space. Uses adaptive thresholding with optional morphological
    cleaning for robust wall detection.
    
    Args:
        image (str | bytes): Floor plan image file path or raw image bytes
        cleaned (bool): Whether to apply morphological operations (default: True)
        
    Returns:
        np.ndarray: Binary mask (uint8) where 1=free space, 0=wall
        
    Image Processing Pipeline:
        1. Load image and convert RGBA to RGB if needed
        2. Convert to grayscale using luminance formula
        3. Apply Otsu's automatic threshold selection
        4. Generate initial binary mask
        5. Optional morphological cleaning:
           - Binary closing to fill small gaps
           - Remove small objects (noise reduction)
           - Binary opening to smooth boundaries
           
    Example:
        ```python
        # From file path
        mask = create_img_mask("floorplan.png")
        
        # From bytes (e.g., uploaded file)
        with open("plan.jpg", "rb") as f:
            mask = create_img_mask(f.read())
        ```
        
    Note:
        Otsu thresholding automatically determines optimal threshold for
        separating walls (dark) from floors (light). Morphological cleaning
        improves mask quality but may alter fine details.
    """
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
    """
    Generate edge-based binary mask using Canny edge detection.
    
    Alternative to Otsu thresholding that detects wall boundaries using
    gradient-based edge detection. Useful for floor plans with unclear
    contrast or complex textures.
    
    Args:
        image (str | bytes): Floor plan image file path or raw bytes
        
    Returns:
        np.ndarray: Binary edge mask (uint8) where 1=edge, 0=no edge
        
    Algorithm:
        1. Convert image to grayscale
        2. Apply Gaussian smoothing (σ=2) for noise reduction
        3. Compute image gradients using Sobel operators
        4. Apply non-maximum suppression
        5. Use double thresholding and edge tracking
        
    Example:
        ```python
        edge_mask = create_canny_img_mask("complex_floorplan.png")
        ```
        
    Note:
        Canny detection focuses on edges rather than filled regions.
        May require post-processing to create solid wall masks.
    """
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
    """
    Save signal heatmap as PNG image file.
    
    Exports heatmap data as a visual PNG image using 'hot' colormap for
    signal strength visualization.
    
    Args:
        data (np.ndarray): 2D heatmap data (normalized 0-1)
        name (str): Base filename for output image
        
    Example:
        ```python
        save_heatmap_to_png(heatmap_data, "sensor_001_heatmap")
        # Saves: src/model_data/sensor_001_heatmap.png
        ```
    """
    plt.imshow(data, cmap='hot')
    plt.title("heatmap")
    plt.axis('off')
    plt.savefig(f"src/model_data/{name}.png")

def fit_calibration_data(sensor_matrix: np.ndarray, position_matrix: np.ndarray)-> Dict[str,float]:
    """
    Fit RSSI-to-distance model parameters using calibration measurements.
    
    Optimizes path loss model parameters (RSSI_0, N) using least squares fitting
    on known distance-RSSI measurement pairs from sensor calibration data.
    
    Args:
        sensor_matrix (np.ndarray): RSSI measurements between sensors
        position_matrix (np.ndarray): Known distances between sensors (same shape)
        
    Returns:
        Dict[str, float]: Optimized parameters {"rssi_0": value, "N": value}
        
    Optimization Process:
        1. Extract valid (non-zero, off-diagonal) RSSI-distance pairs
        2. Use scipy.optimize.curve_fit with bounded parameters:
           - RSSI_0: [-100, -10] dBm (reference signal strength)
           - N: [0, 10] (path loss exponent)
        3. Save parameter covariance matrix for uncertainty analysis
        
    Mathematical Model:
        distance = 10^((RSSI_0 - RSSI) / (10 * N))
        
    Example:
        ```python
        # Calibration data: 3x3 matrices
        rssi_data = np.array([
            [0, -65, -70],
            [-65, 0, -68],
            [-70, -68, 0]
        ])
        positions = np.array([
            [0, 2.5, 3.2],
            [2.5, 0, 2.8],
            [3.2, 2.8, 0]
        ])
        
        params = fit_calibration_data(rssi_data, positions)
        # Returns: {"rssi_0": -52.1, "N": 2.45}
        ```
        
    Files Generated:
        - src/model_data/params.npy: Optimized [RSSI_0, N] parameters
        - src/model_data/params_covariance.npy: Parameter uncertainty matrix
        
    Raises:
        TypeError: If matrix shapes don't match
        
    Note:
        Requires at least 3 data points for stable fitting. More calibration
        measurements improve parameter accuracy and reduce uncertainty.
    """
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
    """
    Compute device importance rankings using heuristic algorithms.
    
    Analyzes sensor data to score device importance based on signal quality,
    detection frequency, sensor coverage, and temporal patterns. Uses advanced
    heuristic models to provide actionable device rankings.
    
    Args:
        all_sensor_data (List[SensorData]): Complete historical sensor dataset
        num_sensors (int): Total number of sensors in system (for normalization)
        selected_keys (Optional[List[str]]): Device MACs to analyze (None = all devices)
        
    Returns:
        Dict[str, float]: Device importance scores {device_mac: score}
        
    Scoring Factors:
        - Signal Quality: Average RSSI strength with temporal decay weighting
        - Hit Frequency: Number of detections with soft thresholding
        - Sensor Coverage: How many different sensors detect the device
        - Temporal Patterns: Recent activity weighted higher than old data
        
    Score Range: 0.0 (never present/low importance) to 1.0 (always present/high importance)
    
    Example:
        ```python
        rankings = target_heuristic(sensor_data, num_sensors=3)
        # Returns: {"device_001": 0.85, "device_002": 0.62, "device_003": 0.31}
        
        # Analyze specific devices only
        subset_rankings = target_heuristic(
            sensor_data, 3, selected_keys=["device_001", "device_002"]
        )
        ```
        
    Algorithm Details:
        1. Parse sensor data into device-centric hit lists by sensor
        2. For each device, collect all (timestamp, RSSI) measurements per sensor
        3. Apply heuristic ranking algorithm with current timestamp
        4. Normalize scores using system parameters (max sensors, thresholds)
        
    Note:
        Uses _rank_target() from heuristic_models for core scoring logic.
        Scores are relative to system configuration and historical patterns.
    """
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
    """
    Train and optimize heuristic model parameters using user ratings.
    
    Machine learning optimization to find optimal heuristic parameters that
    best match user-provided device importance ratings. Uses scipy optimization
    to minimize prediction error on training data.
    
    Args:
        all_sensor_data (List[SensorData]): Historical sensor measurements
        S_max (int): Maximum number of sensors in system
        
    Returns:
        scipy.optimize.OptimizeResult: Optimization results with best parameters
        
    Optimized Parameters:
        - rssi_ref (float): Reference RSSI for signal quality scoring [-80, -40] dBm
        - rssi_min (float): Minimum detectable RSSI [-120, -70] dBm  
        - N_thresh (int): Hit count threshold for importance [1, 20]
        - half_life (float): Temporal decay half-life [100, 100000] seconds
        
    Training Process:
        1. Load user ratings from ratings_log.json
        2. Prepare training tuples: (sensor_hits_by_device, user_rating)
        3. Define objective function: MSE between predicted and actual ratings
        4. Use L-BFGS-B bounded optimization to find best parameters
        5. Return optimization result with parameter values and convergence info
        
    Example:
        ```python
        result = train_heuristic_params(sensor_data, S_max=3)
        if result.success:
            optimal_params = result.x
            print(f"RSSI_ref: {optimal_params[0]:.1f} dBm")
            print(f"N_thresh: {optimal_params[2]:.0f} hits")
        ```
        
    Requirements:
        - User ratings must exist in src/model_data/ratings_log.json
        - Each line: {"device": "mac_address", "rating": 0.0-1.0}
        - Need sufficient rated devices for stable optimization
        
    Note:
        Training effectiveness depends on rating quality and data coverage.
        Requires diverse device behaviors in training data for generalization.
    """
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

