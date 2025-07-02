from fastapi import FastAPI, HTTPException, Request
from fastapi.security import oauth2
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from . models import SensorData
from typing import Dict, List
import sqlite3, json, urllib.parse
from datetime import datetime
from . analyze_data import create_plot_from_SensorData


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

@app.post("/data")
async def handle_data(sensor_data: SensorData):
    if not sensor_data.meta or not sensor_data.results:
        raise HTTPException(status_code=400, detail="Invalid data format")
    else:
        status = _validate_data(sensor_data)
        return {"status": status, "data": sensor_data}

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

def _parse_urls(list):
    return [urllib.parse.unquote(name) for name in list]

def _parse_sensor_row(row):
    meta: Dict[str,str] = json.loads(row[1])      # JSON-merkkijono -> dict
    results: Dict[str,float] = json.loads(row[2])    # JSON-merkkijono -> dict
    return SensorData(meta=meta,results=results)


def _validate_data(sensor_data: SensorData):
    try:
        con, cur = _get_db()
        # Convert lists/dicts to JSON strings for SQLite storage
        current_time = datetime.now().strftime("%Y/%m/%d:%H:%M:%S")
        sensor_data.meta["time"] = current_time
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

