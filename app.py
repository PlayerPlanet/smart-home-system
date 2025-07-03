from fastapi import FastAPI, HTTPException, Request
from fastapi.security import oauth2
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from . models import SensorData, Device
from typing import Dict, List, Set, Tuple
import sqlite3, json, urllib.parse
from datetime import datetime
from . analyze_data import create_plot_from_SensorData, update_distance_parameters
import numpy as np

#fastapi app
app = FastAPI()
app.mount("/img", StaticFiles(directory="img"), name="img")
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

#Known sensors:
sensors: Set[Device] = set()
#Calibration call:
calibrate: str = ""
calibrated_responses: Set[Device] = set()
calibrated_devices: Set[Device] = set()
rssi_results: np.ndarray = None
#TODO: add calibrate call added to return when UI sends get to certain endpoint
@app.post("/data")
async def handle_data(sensor_data: SensorData):
    global calibrate
    if not sensor_data.meta or not sensor_data.results:
        raise HTTPException(status_code=400, detail="Invalid data format")
    else:
        status = _validate_data(sensor_data)
        return {"status": status, "calibrate":calibrate}

@app.get("/data/")
async def load_data():
    data = _fetch_db_data()
    return {"sensor_data":data}

@app.get("/token")
async def token_api():
    token = _create_OAuth_token()
    return {"Authorization":"Bearer","token": token}

@app.get("/devices")
async def load_devices():
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

@app.post("/images")
async def regenerate_images(device_names: List[str]):
    device_names = _parse_urls(device_names)
    all_data = _fetch_db_data()
    parsed_data = [_parse_sensor_row(row) for row in all_data]
    image_paths = create_plot_from_SensorData(parsed_data, device_names)
    return {"image_paths": image_paths}

@app.post("/calibrate")
async def calibrate_distance(sensor_data: SensorData):
    global sensors, rssi_results, calibrated_devices, calibrated_responses, calibrate
    mac = sensor_data.meta.get("mac")
    if mac:
        new_device = Device(mac=mac)
        calibrated_responses.add(new_device) 
    for target_mac, rssi in sensor_data.results.items():
        new_target = Device(mac=target_mac)
        if new_target in sensors:
            i, j = (list(sensors).index(new_device),list(sensors).index(new_target))
            target = new_target
            avg_rssi = rssi_results[i][j]
            rssi_results[i][j] = (avg_rssi+rssi)/2
    if calibrated_responses == sensors:
        calibrated_responses.clear()
        calibrated_devices.add(target)
        devices_left = sensors.difference(calibrated_devices)
        if devices_left:
            next_device = list(devices_left)[0]
            return {"status": f"{len(calibrated_devices)} / {len(sensors)}","calibrate":next_device.mac}
        else:
            status = update_distance_parameters(rssi_results)
            rssi_results.clear()
            calibrated_devices.clear()
            return{"status":status}
    else:
        return {"status": f"{len(calibrated_devices)} / {len(sensors)}","calibrate":calibrate}
    
@app.get("/calibrate", response_class=RedirectResponse)
async def start_calibration(request: Request):
    global calibrate
    if not sensors:
        calibrate = "unknown"
    else:
        calibrate = list(sensors)[0].mac
    return RedirectResponse("/", status_code=200)


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
        with Device(mac=mac) as new_device:
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

