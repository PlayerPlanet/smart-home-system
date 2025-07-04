import matplotlib.pyplot as plt
from .models import SensorData, Device
from .rssi_models import base_model
from typing import List, Dict, Optional, Set, Tuple
from datetime import datetime
import matplotlib.dates as mdates
import numpy as np
from scipy.linalg import eigh

def create_plot_from_SensorData(
    all_sensor_data: List[SensorData], 
    selected_keys: Optional[List[str]] = None,
    output_dir: str = "img/"
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


def _save_plot_to_img(coords: np.ndarray):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(coords[:, 0], coords[:, 1])
    for i in range(len(coords)):
        ax.text(coords[i, 0], coords[i, 1], f'Node {i+1}', fontsize=12)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.savefig("img/distance_plot.png")
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