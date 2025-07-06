import numpy as np
from typing import Dict, Tuple
import matplotlib.pyplot as plt
import math
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def base_model(rssi,rssi_0=-30,N=2):
    return 10 ** ((rssi_0-rssi)/10*N)

def rssi_from_distance(d, rssi_0=-30,N=2):
    return rssi_0-10*N*np.log10(d)

class FloorModel:
    def __init__(self,
                 mask: np.ndarray,
                 scale: float,
                 sensor_positions_m: Dict[str, Tuple[float, float]],
                 image_path: str = None,
                 damping_factor: float = 2.0):
        self.mask = mask.astype(np.uint8)  # seinämaski: 1 = seinä, 0 = vapaa
        self.scale = scale                # metriä per pikseli
        self.sensor_positions_m = sensor_positions_m  # {id: (x,y) metreinä}
        self.image_path = image_path      # alkuperäinen kuva
        # Sisäisesti: myös pikselikoordinaatit
        self.sensor_positions_px = {
            k: (int(x / scale), int(y / scale))
            for k, (x, y) in sensor_positions_m.items()
        }
        self.damping_factor = damping_factor
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
    def show_mask_as_plot(self):
        """näyttää maskin matplotlib.plt objektina"""
        plt.imshow(self.mask, cmap="gray")
        plt.title("Binary Mask")
        plt.axis('off')
        plt.show()
    def save_mask_as_png(self, filename):
        """tallentaa maskin png-kuvana"""
        plt.imshow(self.mask, cmap="gray")
        plt.title("Binary Mask")
        plt.axis('off')
        plt.savefig(f"static/img/binarymask_{filename}.png")
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
    def generate_signal_heatmap(
        self,
        sensor_pos_m: Tuple[float, float],
        max_range_m: float,
        resolution: float = 0.05,
        angle_step_deg: float = 1.0
    ) -> np.ndarray:
        """luo heatmapin alkaen sensor_posista."""
        heatmap = np.zeros_like(self.mask, dtype=np.float32)
        visited = np.zeros_like(self.mask, dtype=bool)
        origin_px = np.array(sensor_pos_m) / self.scale
        origin_px = origin_px.astype(int)

        num_rays = int(360 / angle_step_deg)
        for i in range(num_rays):
            angle = np.deg2rad(i * angle_step_deg)
            self._cast_ray(
                origin_px,
                angle,
                heatmap,
                visited,
                max_range_px=int(max_range_m / self.scale),
                initial_strength=1.0
            )

        # Normalize to [0, 1] for compatibility with scikit-image pipelines
        if heatmap.max() > 0:
            heatmap /= heatmap.max()

        return heatmap

    def _cast_ray(
        self,
        origin_px: np.ndarray,
        angle: float,
        heatmap: np.ndarray,
        visited: np.ndarray,
        max_range_px: int,
        initial_strength: float,
        min_strength: float = 0.01,
        max_bounces: int = 3
    ):
        x, y = origin_px.astype(float)
        dx, dy = math.cos(angle), math.sin(angle)
        step_size = 1.0  # pixels

        strength = initial_strength
        bounces = 0
        path = []

        for _ in range(max_range_px):
            ix, iy = int(round(x)), int(round(y))
            if 0 <= ix < self.mask.shape[1] and 0 <= iy < self.mask.shape[0]:
                if self.mask[iy, ix] == 0:  # Wall hit
                    if bounces >= max_bounces or strength < min_strength:
                        break
                    # Reflect: simple horizontal mirror
                    angle = math.pi - angle
                    dx, dy = math.cos(angle), math.sin(angle)
                    bounces += 1
                    continue

                if not visited[iy, ix]:
                    heatmap[iy, ix] += strength
                    visited[iy, ix] = True

                distance_m = np.linalg.norm([x - origin_px[0], y - origin_px[1]]) * self.scale
                attenuation = rssi_from_distance(distance_m)
                strength = min(initial_strength, attenuation)

                x += dx * step_size
                y += dy * step_size
            else:
                break  # Out of bounds
def wall_interfence_model(rssi, floor_model: FloorModel):
    base_output = base_model(rssi)

import numpy as np
from scipy.ndimage import laplace
from typing import Tuple

def simulate_wave_heatmap(sensor_pos_m: Tuple[float, float],
                          mask: np.ndarray,
                          scale: float,
                          num_steps: int = 500,
                          wave_speed: float = 0.3,
                          dt: float = None,
                          damping: float = 0.01,
                          absorption: float = 0.3) -> np.ndarray:
    """
    Simulates a damped 2D wave from a source and accumulates energy into a heatmap.

    Parameters:
    - sensor_pos_m: (x, y) position of the sensor in meters
    - mask: 2D binary array (1 = free space, 0 = wall)
    - scale: meters per pixel (spatial resolution)
    - num_steps: number of simulation time steps
    - wave_speed: propagation speed in meters per time step
    - dt: time step duration (can be 1.0 if c is scaled)
    - damping: damping factor per step (0-0.1)
    - absorption: wall absorption (0 = full reflect, 1 = full absorption)

    Returns:
    - energy: normalized energy heatmap (values in [0, 1])
    """

    # Derived constants
    dx = scale  # spatial resolution in meters
    if dt is None:
        dt = 0.7 * dx / wave_speed  # auto-select safe dt
    H, W = mask.shape
    u_prev = np.zeros((H, W), dtype=np.float32)
    u_curr = np.zeros((H, W), dtype=np.float32)
    energy = np.zeros((H, W), dtype=np.float32)
    pulse_freq = 1_000  # Hz (you can tune this)
    pulse_duration = 30
    pulse_center = pulse_duration / 2
    sigma = pulse_duration / 6
    # Source position in pixels
    sx, sy = int(sensor_pos_m[0] / scale), int(sensor_pos_m[1] / scale)
    if 0 <= sy < H and 0 <= sx < W:
        u_curr[sy, sx] = 1.0  # initial delta pulse

    # Simulation loop
    for step in range(num_steps):
        # Compute Laplacian
        # Add a few cycles of oscillation at the source to inject real energy
        if step % 100 == 0:
            logger.info(f"Wave simulation step {step}/{num_steps}")
        if step < pulse_duration:
            u_curr[sy, sx] += np.sin(2 * np.pi * pulse_freq * step * dt) * np.exp(-((step - pulse_center) ** 2) / (2 * sigma ** 2))
        lap = laplace(u_curr, mode='constant', cval=0.0)
        coeff = (wave_speed * dt / dx) ** 2
        damp = 1.0 - damping

        # Wave update
        u_next = (2 * u_curr - u_prev) * damp + coeff * lap

        # Walls attenuate the wave
        wall_mask = (mask == 0)
        u_next[wall_mask] *= (1.0 - absorption)
        # Fully absorb wave at the outer boundary (1-pixel thick frame)
        u_next[0, :] = 0.0
        u_next[-1, :] = 0.0
        u_next[:, 0] = 0.0
        u_next[:, -1] = 0.0
        # Accumulate energy (squared amplitude)
        energy += u_next**2

        # Advance frames
        u_prev, u_curr = u_curr, u_next

    # Normalize energy
    energy = np.clip(energy / energy.max(), 0.0, 1.0)

    return energy
