import matplotlib.pyplot as plt
from . models import SensorData
from typing import List, Dict, Optional
import  urllib.parse

def create_plot_from_SensorData(
    all_sensor_data: List[SensorData], 
    selected_keys: Optional[List[str]] = None,
    output_dir: str = "img/"
) -> List[str]:
    t = []
    devices: Dict[str, List[float]] = {}
    for sensor_data in all_sensor_data:
        for key, value in sensor_data.results.items():
            if selected_keys is None or key in selected_keys:
                devices.setdefault(key, []).append(value)
                t.append(sensor_data.meta.get("time"))
    image_paths = []
    for key, value in devices.items():
        plt.plot(t,value,marker="o")
        plt.title(key)
        plt.xlabel("Time")
        plt.ylabel("rssi")
        key = urllib.parse.unquote(key)
        print(key)
        key = key.replace("%3","")
        filename = f"plot_{key}.png"
        plt.savefig(output_dir+filename)
        plt.close()
        image_paths.append(output_dir+filename)
    return image_paths