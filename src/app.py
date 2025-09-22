"""
Smart Home IoT System - Main Application Module

This module implements the core FastAPI web application for the smart home IoT system.
It provides REST API endpoints for sensor data collection, device monitoring, floor plan
configuration, and system calibration.

The application handles:
- RSSI sensor data ingestion and storage
- Device discovery and tracking
- Interactive floor plan configuration  
- Sensor network calibration
- Real-time visualization generation
- Device importance ranking and rating

Key Components:
- FastAPI app with REST endpoints
- SQLite database for time-series data storage
- Calibration management system
- Static file serving for web interface

Author: Smart Home IoT Team
Version: 1.0.0
"""

from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.security import oauth2
from fastapi.responses import HTMLResponse, RedirectResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from .models import SensorData, Device, Rating
from .rssi_models import FloorModel, simulate_wave_heatmap
from typing import Dict, List, Set, Tuple, BinaryIO
import sqlite3, json, urllib.parse
from datetime import datetime
from .analyze_data import create_plot_from_SensorData, update_distance_parameters, create_img_mask, save_heatmap_to_png, target_heuristic
import numpy as np
import os

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#fastapi app
app = FastAPI()
app.mount("/static/", StaticFiles(directory="static/"), name="static")
templates = Jinja2Templates(directory="templates")
#sqlite3 db
def _get_db():
    """
    Initialize and return SQLite database connection with cursor.
    
    Creates the sensor_data table if it doesn't exist. The table schema:
    - id: Primary key (auto-increment integer)
    - meta: JSON string containing sensor metadata (MAC, timestamp, etc.)
    - results: JSON string containing RSSI measurements by device MAC
    
    Returns:
        tuple[sqlite3.Connection, sqlite3.Cursor]: Database connection and cursor
        
    Note:
        Uses row factory for dict-like access to query results
    """
    con = sqlite3.connect("sensor_data.db")
    cur = con.cursor()
    con.row_factory = sqlite3.Row
    cur.execute("""
        CREATE TABLE IF NOT EXISTS sensor_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            meta TEXT,
            results TEXT
        )
    """)
    return con, cur
con, _ = _get_db()
con.commit()

#Calibration call:
class CalibrationManager:
    """
    Manages the sensor calibration process for distance parameter optimization.
    
    The calibration process involves:
    1. Collecting sensors that participate in calibration
    2. Sequentially requesting RSSI measurements from each sensor
    3. Building a complete RSSI matrix between all sensor pairs
    4. Computing optimal distance model parameters using optimization
    
    Attributes:
        sensors (Set[Device]): Set of participating sensor devices
        calibrate (str): MAC address of currently calibrating sensor (empty if not calibrating)
        calibrated_responses (Set[Device]): Sensors that have responded in current round
        calibrated_devices (Set[Device]): Sensors that have completed all measurements
        rssi_results (np.ndarray): Matrix storing RSSI values between sensor pairs
    """
    def __init__(self):
        """Initialize calibration manager with empty state."""
        self.sensors: Set[Device] = set()
        self.calibrate: str = ""
        self.calibrated_responses: Set[Device] = set()
        self.calibrated_devices: Set[Device] = set()
        self.rssi_results: np.ndarray = None

    def add_sensor(self, device: Device):
        """
        Add a new sensor to the calibration set.
        
        Args:
            device (Device): Sensor device to add to calibration
            
        Note:
            Initializes RSSI results matrix when first sensor is added
        """
        if device not in self.sensors:
            self.sensors.add(device)
            self.rssi_results = np.zeros([len(self.sensors), len(self.sensors)])

    def start(self):
        """
        Begin calibration process with the first available sensor.
        
        Sets the calibrate field to the MAC address of the first sensor,
        signaling that this sensor should begin transmitting measurements.
        """
        if not self.sensors:
            self.calibrate = ""
        else:
            self.calibrate = list(self.sensors)[0].mac

    def stop(self):
        """
        Stop calibration process and reset all state.
        
        Clears all calibration data including sensor sets, RSSI matrix,
        and current calibration status.
        """
        self.calibrate = ""
        self.calibrated_devices.clear()
        self.calibrated_responses.clear()
        self.sensors.clear()
        if self.rssi_results is not None:
            self.rssi_results = np.zeros_like(self.rssi_results)

    def handle_response(self, sensor_data: SensorData):
        """
        Process calibration response from a sensor.
        
        Handles incoming RSSI measurements during calibration, updating the
        results matrix and managing the calibration state machine.
        
        Args:
            sensor_data (SensorData): RSSI measurements from sensor
            
        Returns:
            dict: Status update with current progress and next sensor to calibrate
            
        The calibration flow:
        1. Extract MAC and validate sensor
        2. Store RSSI measurements in results matrix  
        3. Check if all sensors have responded for current round
        4. Move to next sensor or complete calibration
        5. When complete, optimize distance parameters and save results
        """
        mac = sensor_data.meta.get("mac")
        if not mac:
            return {"status": "Missing MAC address", "calibrate": self.calibrate}
        new_device = Device(mac=mac)
        self.calibrated_responses.add(new_device)
        for target_mac, rssi in sensor_data.results.items():
            target_device = Device(mac=target_mac)
            if target_device in self.sensors:
                i, j = (list(self.sensors).index(new_device), list(self.sensors).index(target_device))
                avg_rssi = self.rssi_results[i][j]
                self.rssi_results[i][j] = (avg_rssi + rssi) / 2

        if self.calibrated_responses == self.sensors:
            self.calibrated_responses.clear()
            self.calibrated_devices.add(new_device)
            devices_left = self.sensors.difference(self.calibrated_devices)
            if devices_left:
                next_device = list(devices_left)[0]
                self.calibrate = next_device.mac
                return {"status": f"{len(self.calibrated_devices)} / {len(self.sensors)}", "calibrate": self.calibrate}
            else:
                np.save("/model_data/calibration_results.npy",self.rssi_results)
                status = update_distance_parameters(self.rssi_results)
                self.rssi_results = np.zeros_like(self.rssi_results)
                self.calibrated_devices.clear()
                self.calibrate = ""
                return {"status": status}
        else:
            return {"status": f"{len(self.calibrated_devices)} / {len(self.sensors)}", "calibrate": self.calibrate}

calibration = CalibrationManager()
class SettingsManager:
    """
    Placeholder class for future settings management functionality.
    
    Currently unused but designed to handle:
    - Configuration file management
    - Runtime settings updates
    - Settings validation and persistence
    
    Attributes:
        new_settings (bool): Flag indicating new settings are available
        settings_path (str): Path to settings configuration file
        settings_format (type): Expected format for settings (JSON)
    """
    def __init__(self):
        self.new_settings: bool = False
        self.settings_path: str = "templates/settings.json"
        self.settings_format: type = json

#TODO: add calibrate call added to return when UI sends get to certain endpoint
@app.post("/data")
async def handle_data(sensor_data: SensorData):
    """
    Receive and store RSSI sensor data from IoT sensors.
    
    Primary endpoint for sensor data ingestion. Validates incoming data,
    stores it in the database, and manages sensor registration for calibration.
    
    Args:
        sensor_data (SensorData): RSSI measurements and metadata from sensor
        
    Returns:
        dict: Response containing processing status and calibration instructions
        
    Raises:
        HTTPException: 400 if data format is invalid (missing meta or results)
        
    Example:
        POST /data
        {
            "meta": {"mac": "sensor_001", "time": "2025/01/15:14:30:00"},
            "results": {"device_001": -65.2, "device_002": -72.8}
        }
        
        Response: {"status": "success", "calibrate": "sensor_002"}
    """
    if not sensor_data.meta or not sensor_data.results:
        raise HTTPException(status_code=400, detail="Invalid data format")
    else:
        status = _validate_data(sensor_data)
        mac = sensor_data.meta.get("mac")
        if mac:
            logger.info(f"Received data from MAC: {mac}")
            calibration.add_sensor(Device(mac=mac))
        return {"status": status, "calibrate":calibration.calibrate}

@app.get("/data/")
async def load_data():
    """
    Retrieve all historical sensor data from the database.
    
    Returns complete dataset of all RSSI measurements collected by the system.
    Used for analysis, visualization, and debugging purposes.
    
    Returns:
        dict: All sensor data records with metadata and RSSI measurements
        
    Example Response:
        {
            "sensor_data": [
                ["1", "{\"mac\": \"sensor_001\", \"time\": \"2025/01/15:14:30:00\"}", 
                 "{\"device_001\": -65.2, \"device_002\": -72.8}"],
                ...
            ]
        }
    """
    data = _fetch_db_data()
    return {"sensor_data":data}

@app.get("/token")
async def token_api():
    """
    Retrieve OAuth authentication token (placeholder implementation).
    
    Currently returns None as authentication is not implemented.
    Reserved for future OAuth/JWT token-based authentication system.
    
    Returns:
        dict: Authentication token information
        
    Example Response:
        {"Authorization": "Bearer", "token": null}
    """
    token = _create_OAuth_token()
    return {"Authorization":"Bearer","token": token}

@app.get("/sensors")
async def load_sensors():
    """
    Get list of active sensors registered with the system.
    
    Returns MAC addresses of all sensors that have submitted data or are
    participating in calibration. In TEST mode, returns mock sensor names.
    
    Returns:
        dict: List of sensor MAC addresses or names
        
    Example Response:
        {"sensors": ["sensor_001", "sensor_002", "sensor_003"]}
        
    Note:
        In test mode (TEST=true), returns predefined sensor names for development
    """
    macs = [sensor.mac for sensor in list(calibration.sensors)]
    if not macs and os.environ.get("TEST",True):
        macs = ["first_sensor","second_sensor", "third_sensor"]
    return {"sensors": macs}

@app.get("/devices")
async def load_devices():
    """
    Get all detected devices ranked by importance/presence.
    
    Extracts unique device MAC addresses from sensor data, computes importance
    rankings using heuristic algorithms, and returns devices sorted by score.
    
    Returns:
        dict: Device MAC addresses sorted by importance (highest first)
        
    Example Response:
        {"devices": ["device_001", "device_003", "device_002"]}
        
    Note:
        Ranking considers signal strength, detection frequency, sensor coverage,
        and temporal patterns to determine device importance
    """
    res = []
    try:
        con, cur = _get_db()
        res = cur.execute("SELECT results FROM sensor_data").fetchall()
    finally:
        con.close()
    device_names = set()
    for row in res:
        results_json = row[0]
        try:
            results = json.loads(results_json)
            device_names.update(results.keys())
        except Exception:
            continue
    all_data = _fetch_db_data()
    parsed_data = [_parse_sensor_row(row) for row in all_data]
    rank = target_heuristic(parsed_data, 5, device_names)
    sorted_devices = sorted(device_names, key=lambda mac: rank.get(mac, 0.0), reverse=True)
    return {"devices": sorted_devices}

@app.get("/", response_class=HTMLResponse)
async def read_index(request: Request):
    """
    Serve the main dashboard HTML page.
    
    Returns the primary user interface for device monitoring, sensor status,
    and system control. Provides access to device selection, plotting, and
    calibration controls.
    
    Args:
        request (Request): FastAPI request object for template context
        
    Returns:
        TemplateResponse: Rendered index.html template
    """
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/floor", response_class=HTMLResponse)
async def read_index(request: Request):
    """
    Serve the floor plan configuration interface.
    
    Provides interactive floor plan upload, sensor positioning, scale calibration,
    and measurement tools. Allows users to configure the physical layout of
    their sensor network.
    
    Args:
        request (Request): FastAPI request object for template context
        
    Returns:
        TemplateResponse: Rendered floorplan.html template
    """
    return templates.TemplateResponse("floorplan.html", {"request": request})

@app.post("/floor")
async def upload(image: UploadFile = File(...), metadata: str = Form(...)):
    """
    Process uploaded floor plan image and sensor configuration.
    
    Handles floor plan image upload, extracts sensor positions and scale from
    metadata, creates binary mask for wall detection, and generates signal
    heatmaps for each sensor using wave simulation.
    
    Args:
        image (UploadFile): Floor plan image file (PNG, JPG, etc.)
        metadata (str): JSON string containing scale and sensor positions
        
    Returns:
        dict: Processing confirmation with file path and metadata
        
    Example metadata:
        {
            "scale": 0.05,
            "sensor_positions": {
                "sensor_001": {"x": 100, "y": 150},
                "sensor_002": {"x": 300, "y": 200}
            }
        }
        
    Note:
        Generates RF propagation heatmaps for each sensor using physics-based
        wave simulation with wall interference modeling
    """
    floor_img_path = os.path.join("static/img/","floorplan.png")
    imageFile = await image.read()
    metadata_dict = json.loads(metadata)
    output = _handle_floorplan(floor_img_path,metadata_dict, imageFile)
    return {
        "received_image": floor_img_path,
        "metadata": metadata_dict
    }

@app.post("/images")
async def regenerate_images(device_names: List[str]):
    """
    Generate time-series plots and rankings for specified devices.
    
    Creates visualization plots showing RSSI signal patterns over time for
    selected devices. Also computes and returns device importance rankings.
    
    Args:
        device_names (List[str]): List of device MAC addresses to plot
        
    Returns:
        dict: Generated image file paths and device rankings
        
    Example:
        POST /images
        ["device_001", "device_002"]
        
        Response:
        {
            "image_paths": ["static/img/plot_device_001.png", "static/img/plot_device_002.png"],
            "rank": {"device_001": 0.85, "device_002": 0.62}
        }
        
    Note:
        Images are saved to static/img/ directory and can be accessed via web interface
    """
    device_names = _parse_urls(device_names)
    all_data = _fetch_db_data()
    parsed_data = [_parse_sensor_row(row) for row in all_data]
    image_paths = create_plot_from_SensorData(parsed_data, device_names)
    rank = target_heuristic(parsed_data, 5, device_names)
    return {"image_paths": image_paths, "rank":rank}

@app.post("/calibrate")
async def calibrate_distance(sensor_data: SensorData):
    """
    Handle calibration data submission from sensors.
    
    Processes RSSI measurements during calibration phase, building the
    inter-sensor distance matrix for parameter optimization.
    
    Args:
        sensor_data (SensorData): RSSI measurements from calibrating sensor
        
    Returns:
        dict: Calibration progress and next sensor instructions
        
    Note:
        Part of the automatic calibration workflow managed by CalibrationManager
    """
    return calibration.handle_response(sensor_data)

@app.get("/calibrate", response_class=RedirectResponse)
async def start_calibration(request: Request):
    """
    Start the sensor calibration process.
    
    Initiates automatic calibration of sensor network to optimize distance
    calculation parameters. Redirects to main dashboard after starting.
    
    Args:
        request (Request): FastAPI request object
        
    Returns:
        RedirectResponse: Redirect to main dashboard (/) with 200 status
        
    Note:
        Calibration continues in background until all sensors complete measurements
    """
    calibration.start()
    return RedirectResponse("/", status_code=200)

@app.get("/calibrate/stop")
async def stop_calibration():
    """
    Stop the calibration process and reset state.
    
    Immediately halts ongoing calibration, clears all calibration data,
    and resets the system to normal operation mode.
    
    Returns:
        dict: Confirmation of calibration stop with current status
        
    Example Response:
        {"status": "Calibration stopped", "calibrate": ""}
    """
    calibration.stop()
    return {"status": "Calibration stopped", "calibrate": calibration.calibrate}

@app.get("/settings")
async def send_settings():
    """
    Retrieve system configuration settings.
    
    Returns the current system settings as a JSON file. Used by frontend
    for configuration display and modification.
    
    Returns:
        FileResponse: settings.json file with current configuration
        
    Note:
        Currently serves static settings file; future versions may support
        dynamic configuration management
    """
    return FileResponse("templates/settings.json", media_type="application/json")

@app.post("/rate")
async def rate_device(rating: Rating):
    """
    Submit user rating for device importance.
    
    Accepts user-provided importance ratings for devices, which are used
    to train and improve the automatic device ranking algorithms.
    
    Args:
        rating (Rating): Device rating with MAC address and score (0.0-1.0)
        
    Returns:
        dict: Confirmation of rating submission
        
    Example:
        POST /rate
        {"device": "device_001", "rating": 0.85}
        
        Response: {"status": "ok"}
        
    Note:
        Ratings are logged to JSON file and used for machine learning training
    """
    # You can write this to DB, CSV, or just print for now
    basepath = os.path.dirname(__file__)
    file_path = os.path.join(basepath, "model_data/ratings_log.json")
    with open(file_path, "a") as f:
        f.write(rating.model_dump_json() + "\n")
    return {"status": "ok"}

@app.get("/rate", response_class=HTMLResponse)
async def read_index(request: Request):
    """
    Serve the device rating interface.
    
    Provides interactive interface for users to rate device importance.
    Shows device plots and allows slider-based rating input for training
    the machine learning algorithms.
    
    Args:
        request (Request): FastAPI request object for template context
        
    Returns:
        TemplateResponse: Rendered rate_devices.html template
    """
    return templates.TemplateResponse("rate_devices.html", {"request": request})


#Helper functions#

def _parse_urls(list):
    """
    Decode URL-encoded strings in a list.
    
    Args:
        list (List[str]): List of potentially URL-encoded strings
        
    Returns:
        List[str]: List with URL-decoded strings
    """
    return [urllib.parse.unquote(name) for name in list]

def _parse_sensor_row(row):
    """
    Parse database row into SensorData object.
    
    Converts database row with JSON strings back into structured SensorData
    object for processing and analysis.
    
    Args:
        row: Database row with meta and results JSON strings
        
    Returns:
        SensorData: Parsed sensor data object
        
    Note:
        Expects row[1] to contain meta JSON and row[2] to contain results JSON
    """
    meta: Dict[str,str] = json.loads(row[1])      # JSON-merkkijono -> dict
    results: Dict[str,float] = json.loads(row[2])    # JSON-merkkijono -> dict
    return SensorData(meta=meta,results=results)


def _validate_data(sensor_data: SensorData):
    """
    Validate and store sensor data in database.
    
    Processes incoming sensor data by adding timestamp, registering new sensors,
    converting to JSON format, and storing in SQLite database.
    
    Args:
        sensor_data (SensorData): Validated sensor data to store
        
    Returns:
        str: "success" if data stored successfully
        
    Note:
        Automatically adds current timestamp to metadata and updates
        global sensor registry for calibration purposes
    """
    global rssi_results, sensors
    try:
        con, cur = _get_db()
        # Convert lists/dicts to JSON strings for SQLite storage
        current_time = datetime.now().strftime("%Y/%m/%d:%H:%M:%S")
        sensor_data.meta["time"] = current_time
        mac = sensor_data.meta.get("mac")
        new_device = Device(mac=mac) 
        if mac and new_device not in sensors:
            sensors.add(new_device) 
            rssi_results = np.zeros([len(sensors),len(sensors)])
        meta_str = json.dumps(sensor_data.meta)
        results_str = json.dumps(sensor_data.results)
        
        cur.execute("""
            INSERT INTO sensor_data (meta, results)
            VALUES (?, ?)
        """, (meta_str, results_str))
        
        # Commit the transaction
        con.commit()
    finally:
        con.close()
        return "success"
    
def _handle_floorplan(floor_img_path: str, metadata_dict, imageFile: BinaryIO):
    """
    Process uploaded floor plan and generate sensor heatmaps.
    
    Handles complete floor plan processing workflow:
    1. Save uploaded image to static directory
    2. Extract and format sensor positions from metadata
    3. Create binary mask from image for wall detection
    4. Generate FloorModel with scale and sensor positions
    5. Run wave simulation for each sensor to create heatmaps
    
    Args:
        floor_img_path (str): Path where floor plan image will be saved
        metadata_dict (dict): Scale and sensor position data
        imageFile (BinaryIO): Uploaded image file data
        
    Returns:
        None
        
    Note:
        Generates RF propagation heatmaps using physics-based wave simulation
        with 2000 time steps and minimal damping for realistic results
    """
    basepath = os.path.dirname(__file__)
    floor_model_path = os.path.join(basepath, "model_data/")
    scale = metadata_dict['scale']
    sensor_positions = metadata_dict['sensor_positions']
    sensor_positions_formated: Dict[str, Tuple[float,float]] = {
        key: (float(value['x'])*scale, float(value['y'])*scale)
        for key, value in sensor_positions.items()
    }
    with open(floor_img_path,"wb") as file:
        n = file.write(imageFile)
    mask = create_img_mask(imageFile)
    floor_model = FloorModel(mask, scale, sensor_positions_formated, floor_img_path)
    floor_model.save(floor_model_path)
    for sensor_name, sensor_pos_m in floor_model.sensor_positions_m.items():
        sensor_pos = tuple(map(float, sensor_pos_m))
        energy = simulate_wave_heatmap(sensor_pos, mask, scale,2000, damping=0.0001)
        log_energy = np.log10(energy + 1e-10)  # Avoid log(0)
        log_energy = (log_energy - log_energy.min()) / (log_energy.max() - log_energy.min())
        save_heatmap_to_png(log_energy, "heatmap"+sensor_name+datetime.strftime(datetime.now(), "%Y-%m-%d-%H-%M"))
    return None

def _fetch_db_data():
    """
    Retrieve all sensor data records from database.
    
    Connects to SQLite database and fetches all historical sensor data
    for analysis and processing purposes.
    
    Returns:
        List[sqlite3.Row]: All sensor data records from database
        
    Note:
        Ensures database connection is properly closed after retrieval
    """
    try:
        con, cur = _get_db()
        res = cur.execute("""
            SELECT * FROM sensor_data
        """).fetchall()
    finally:
        con.close()
        return res
    
def _create_OAuth_token():
    """
    Generate OAuth authentication token (placeholder).
    
    Reserved for future implementation of OAuth/JWT-based authentication.
    Currently returns None as authentication is not implemented.
    
    Returns:
        None: Placeholder return value
        
    Note:
        Future implementation should generate secure JWT tokens for API access
    """
    return None

