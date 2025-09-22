# Smart Home IoT System

A comprehensive IoT system for tracking and analyzing wireless devices using RSSI (Received Signal Strength Indicator) signals from multiple sensors. The system provides real-time device localization, signal visualization, and intelligent device ranking for smart home environments.

## 🏠 Overview

This smart home system monitors IoT devices by collecting RSSI measurements from strategically placed sensors throughout your home. It uses advanced signal processing, machine learning, and wave simulation techniques to:

- **Track device locations** using trilateration and signal strength analysis
- **Visualize signal patterns** with interactive heatmaps and plots  
- **Rank devices intelligently** based on presence, signal quality, and temporal patterns
- **Calibrate sensor networks** automatically for optimal accuracy
- **Simulate RF propagation** through floor plans with wall interference modeling

## 🚀 Features

### Core Functionality
- **Real-time RSSI Data Collection**: Capture and store signal strength measurements from multiple sensors
- **Device Discovery & Tracking**: Automatically detect and track IoT devices by MAC address
- **Interactive Floor Plan Mapping**: Upload floor plans and position sensors visually
- **Signal Visualization**: Generate time-series plots and heatmaps for device signals
- **Intelligent Device Ranking**: Score devices based on presence, signal quality, and activity patterns

### Advanced Capabilities  
- **Wave Simulation**: Physics-based RF propagation modeling with wall interference
- **Automatic Calibration**: Self-calibrating sensor networks using optimization algorithms
- **Machine Learning**: Heuristic models for device importance scoring
- **REST API**: Complete FastAPI-based backend for integration
- **Web Interface**: Responsive HTML/CSS/JavaScript frontend

## 🏗️ Architecture

```
smart-home-system/
├── src/                    # Core Python modules
│   ├── app.py             # FastAPI web application & API endpoints
│   ├── models.py          # Pydantic data models
│   ├── rssi_models.py     # Signal processing & wave simulation
│   ├── analyze_data.py    # Data analysis & visualization
│   ├── heuristic_models.py # Device ranking algorithms
│   └── run_training_heuristic.py # Model training script
├── templates/             # HTML templates
│   ├── index.html        # Main dashboard
│   ├── floorplan.html    # Floor plan configuration
│   └── rate_devices.html # Device rating interface
├── static/               # Frontend assets
│   ├── css/styles.css    # Styling
│   ├── js/floorplan.js   # Interactive floor plan editor
│   └── img/              # Generated plots & heatmaps
├── compose.yaml          # Docker deployment
├── Dockerfile           # Container configuration
└── requirements.txt     # Python dependencies
```

### System Components

1. **Data Collection Layer** (`app.py`)
   - FastAPI endpoints for sensor data ingestion
   - SQLite database for time-series storage
   - Real-time device discovery

2. **Signal Processing Layer** (`rssi_models.py`)
   - RSSI-to-distance conversion models
   - RF wave propagation simulation
   - Wall interference calculations

3. **Analysis Layer** (`analyze_data.py`)
   - Time-series visualization
   - Multidimensional scaling (MDS)
   - Calibration parameter optimization

4. **Intelligence Layer** (`heuristic_models.py`)
   - Device importance scoring
   - Temporal pattern analysis
   - Presence detection algorithms

5. **User Interface Layer** (`templates/`, `static/`)
   - Interactive dashboards
   - Floor plan configuration
   - Real-time visualizations

## 📦 Installation

### Prerequisites
- Python 3.10+
- Docker (optional, for containerized deployment)
- Modern web browser with JavaScript support

### Local Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd smart-home-system
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the application**
   ```bash
   uvicorn src.app:app --host 0.0.0.0 --port 5000
   ```

4. **Access the web interface**
   ```
   http://localhost:5000
   ```

### Docker Deployment

1. **Build and run with Docker Compose**
   ```bash
   docker-compose up --build
   ```

2. **Access the application**
   ```
   http://localhost:8000
   ```

## 🎯 Usage Guide

### 1. Sensor Setup & Calibration

1. **Configure Sensors**: Navigate to `/floor` to upload your floor plan and position sensors
2. **Define Scale**: Set the pixel-to-meter ratio by clicking two points on your floor plan
3. **Place Sensors**: Click on the floor plan to position your sensors accurately
4. **Start Calibration**: Click "calibrate devices" to begin automatic sensor calibration

### 2. Data Collection

Sensors should POST data to `/data` endpoint:

```json
{
  "meta": {
    "mac": "sensor_001",
    "time": "2025/01/15:14:30:00"
  },
  "results": {
    "device_001": -65.2,
    "device_002": -72.8,
    "device_003": -58.1
  }
}
```

### 3. Device Monitoring

- **View Devices**: Main dashboard shows all detected devices
- **Select Device**: Choose from dropdown to view signal patterns  
- **View Plots**: Time-series plots show RSSI over time
- **Check Rankings**: Device importance scores appear automatically

### 4. Device Rating & Training

1. Navigate to `/rate` to rate device importance
2. Use the slider to score devices from 0.0 to 1.0
3. System learns from your ratings to improve automatic scoring

## 📡 API Reference

### Data Endpoints

- `POST /data` - Submit sensor RSSI measurements
- `GET /data/` - Retrieve all historical sensor data
- `GET /devices` - List all detected devices with rankings

### Configuration Endpoints  

- `GET /sensors` - List active sensors
- `POST /floor` - Upload floor plan and sensor positions
- `GET /calibrate` - Start sensor calibration
- `POST /calibrate` - Submit calibration measurements

### Visualization Endpoints

- `POST /images` - Generate plots for specific devices
- `GET /rate` - Device rating interface
- `POST /rate` - Submit device importance ratings

### Authentication

- `GET /token` - Retrieve OAuth token (placeholder for future auth)

## 🔬 Technical Details

### Signal Processing

The system uses a logarithmic path loss model for RSSI-to-distance conversion:

```
d = 10^((RSSI_0 - RSSI) / (10 * N))
```

Where:
- `RSSI_0`: Reference signal strength at 1 meter
- `N`: Path loss exponent (environment-dependent)
- `d`: Estimated distance in meters

### Wave Simulation

RF propagation is modeled using a 2D damped wave equation with wall absorption:

```
∂²u/∂t² = c²∇²u - γ∂u/∂t
```

This enables realistic signal heatmap generation accounting for:
- Wall reflections and absorption
- Multi-path propagation effects  
- Distance-based signal attenuation

### Device Ranking Algorithm

Devices are scored using a multi-factor heuristic:

```
score = C_rssi × C_hits × C_sensors × C_decay
```

Factors include:
- **Signal Quality** (`C_rssi`): Average RSSI normalized to [0,1]
- **Hit Count** (`C_hits`): Number of detections with soft thresholding
- **Sensor Coverage** (`C_sensors`): How many sensors detect the device
- **Temporal Decay** (`C_decay`): Exponential decay based on last seen time

## 🛠️ Development

### Running Tests

```bash
# Placeholder - implement tests as needed
python -m pytest tests/
```

### Training Heuristic Models

```bash
python -m src.run_training_heuristic
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes with proper documentation
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Support

For questions, issues, or feature requests, please open an issue on the project repository.

---

**Built with** Python, FastAPI, NumPy, SciPy, Matplotlib, and scikit-image