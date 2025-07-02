import matplotlib.pyplot as plt
from . models import SensorData
from typing import List, Dict, Optional
from datetime import datetime
import matplotlib.dates as mdates

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