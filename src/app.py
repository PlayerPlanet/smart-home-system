from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.security import oauth2
from fastapi.responses import HTMLResponse, RedirectResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from .models import SensorData, Device
from typing import Dict, List, Set, Tuple
import sqlite3, json, urllib.parse
from datetime import datetime
from .analyze_data import create_plot_from_SensorData, update_distance_parameters
import numpy as np
import os

#fastapi app
app = FastAPI()
app.mount("/static/", StaticFiles(directory="static/"), name="static")
templates = Jinja2Templates(directory="templates")
#sqlite3 db
def _get_db():
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
    def __init__(self):
        self.sensors: Set[Device] = set()
        self.calibrate: str = ""
        self.calibrated_responses: Set[Device] = set()
        self.calibrated_devices: Set[Device] = set()
        self.rssi_results: np.ndarray = None

    def add_sensor(self, device: Device):
        if device not in self.sensors:
            self.sensors.add(device)
            self.rssi_results = np.zeros([len(self.sensors), len(self.sensors)])

    def start(self):
        if not self.sensors:
            self.calibrate = ""
        else:
            self.calibrate = list(self.sensors)[0].mac

    def stop(self):
        self.calibrate = ""
        self.calibrated_devices.clear()
        self.calibrated_responses.clear()
        self.sensors.clear()
        if self.rssi_results is not None:
            self.rssi_results = np.zeros_like(self.rssi_results)

    def handle_response(self, sensor_data: SensorData):
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
                status = update_distance_parameters(self.rssi_results)
                self.rssi_results = np.zeros_like(self.rssi_results)
                self.calibrated_devices.clear()
                self.calibrate = ""
                return {"status": status}
        else:
            return {"status": f"{len(self.calibrated_devices)} / {len(self.sensors)}", "calibrate": self.calibrate}

calibration = CalibrationManager()

class SettingsManager:
    def __init__(self):
        self.new_settings: bool = False
        self.settings_path: str = "templates/settings.json"
        self.settings_format: type = json

#TODO: add calibrate call added to return when UI sends get to certain endpoint
@app.post("/data")
async def handle_data(sensor_data: SensorData):
    if not sensor_data.meta or not sensor_data.results:
        raise HTTPException(status_code=400, detail="Invalid data format")
    else:
        status = _validate_data(sensor_data)
        mac = sensor_data.meta.get("mac")
        if mac:
            calibration.add_sensor(Device(mac=mac))
        return {"status": status, "calibrate":calibration.calibrate}

@app.get("/data/")
async def load_data():
    data = _fetch_db_data()
    return {"sensor_data":data}

@app.get("/token")
async def token_api():
    token = _create_OAuth_token()
    return {"Authorization":"Bearer","token": token}

@app.get("/sensors")
async def load_sensors():
    macs = [sensor.mac for sensor in list(calibration.sensors)]
    if not macs and os.environ.get("TEST",True):
        macs = ["first_sensor","second_sensor", "third_sensor"]
    return {"sensors": macs}

@app.get("/devices")
async def load_devices():
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
    return {"devices": list(device_names)}

@app.get("/", response_class=HTMLResponse)
async def read_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/floor", response_class=HTMLResponse)
async def read_index(request: Request):
    return templates.TemplateResponse("floorplan.html", {"request": request})
@app.post("/floor")
async def upload(image: UploadFile = File(...), metadata: UploadFile = File(...)):
    metadata_dict = json.loads(await metadata.read())
    
    return {
        "received_image": image.filename,
        "metadata": metadata_dict
    }
@app.post("/images")
async def regenerate_images(device_names: List[str]):
    device_names = _parse_urls(device_names)
    all_data = _fetch_db_data()
    parsed_data = [_parse_sensor_row(row) for row in all_data]
    image_paths = create_plot_from_SensorData(parsed_data, device_names)
    return {"image_paths": image_paths}

@app.post("/calibrate")
async def calibrate_distance(sensor_data: SensorData):
   return calibration.handle_response(sensor_data)

@app.get("/calibrate", response_class=RedirectResponse)
async def start_calibration(request: Request):
    calibration.start()
    return RedirectResponse("/", status_code=200)

@app.get("/calibrate/stop")
async def stop_calibration():
    calibration.stop()
    return {"status": "Calibration stopped", "calibrate": calibration.calibrate}

@app.get("/settings")
async def send_settings():
    return FileResponse("templates/settings.json", media_type="application/json")





#Helper functions#

def _parse_urls(list):
    return [urllib.parse.unquote(name) for name in list]

def _parse_sensor_row(row):
    meta: Dict[str,str] = json.loads(row[1])      # JSON-merkkijono -> dict
    results: Dict[str,float] = json.loads(row[2])    # JSON-merkkijono -> dict
    return SensorData(meta=meta,results=results)


def _validate_data(sensor_data: SensorData):
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

def _fetch_db_data():
    try:
        con, cur = _get_db()
        res = cur.execute("""
            SELECT * FROM sensor_data
        """).fetchall()
    finally:
        con.close()
        return res
    
def _create_OAuth_token():
    return None

