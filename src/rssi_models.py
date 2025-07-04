import numpy as np
from typing import Dict, Tuple

def base_model(rssi,rssi_0=-30,N=2):
    return 10 ** ((rssi_0-rssi)/10*N)

class FloorModel:
    def __init__(self,
                 mask: np.ndarray,
                 scale: float,
                 sensor_positions_m: Dict[str, Tuple[float, float]],
                 image_path: str = None):
        self.mask = mask.astype(np.uint8)  # seinämaski: 1 = seinä, 0 = vapaa
        self.scale = scale                # metriä per pikseli
        self.sensor_positions_m = sensor_positions_m  # {id: (x,y) metreinä}
        self.image_path = image_path      # alkuperäinen kuva
        
        # Sisäisesti: myös pikselikoordinaatit
        self.sensor_positions_px = {
            k: (int(x / scale), int(y / scale))
            for k, (x, y) in sensor_positions_m.items()
        }
    def get_wall_count(self, p1_m: Tuple[float, float], p2_m: Tuple[float, float]) -> int:
        """
        Laske kuinka monta seinää viiva (p1 → p2) ylittää
        """
        from skimage.draw import line
        
        # Muunna pisteet metreistä pikseleihin
        p1_px = (int(p1_m[1] / self.scale), int(p1_m[0] / self.scale))  # (y, x)
        p2_px = (int(p2_m[1] / self.scale), int(p2_m[0] / self.scale))

        rr, cc = line(*p1_px, *p2_px)
        rr = np.clip(rr, 0, self.mask.shape[0]-1)
        cc = np.clip(cc, 0, self.mask.shape[1]-1)
        return int(np.sum(self.mask[rr, cc]))

    def save(self, folder: str):
        """Tallenna maski ja koordinaatit"""
        import os
        np.save(os.path.join(folder, "mask.npy"), self.mask)
        np.save(os.path.join(folder, "sensor_positions_m.npy"), self.sensor_positions_m)
        with open(os.path.join(folder, "scale.txt"), 'w') as f:
            f.write(str(self.scale))
        if self.image_path:
            with open(os.path.join(folder, "image_path.txt"), 'w') as f:
                f.write(self.image_path)

    @staticmethod
    def load(folder: str):
        """Lataa FloorModel tiedostoista"""
        import os
        mask = np.load(os.path.join(folder, "mask.npy"))
        sensor_positions = np.load(os.path.join(folder, "sensor_positions_m.npy"), allow_pickle=True).item()
        with open(os.path.join(folder, "scale.txt"), 'r') as f:
            scale = float(f.read())
        try:
            with open(os.path.join(folder, "image_path.txt"), 'r') as f:
                image_path = f.read()
        except FileNotFoundError:
            image_path = None
        return FloorModel(mask, scale, sensor_positions, image_path)