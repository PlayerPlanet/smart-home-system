// floorplan.js — merged and extended version

const canvas = document.getElementById('floorplan-img');
const ctx = canvas.getContext('2d');
const point1 = document.getElementById('point1');
const point2 = document.getElementById('point2');
const scaleText = document.getElementById('scaleText');
const sensorList = document.getElementById('sensor-list');
const saveBtn = document.getElementById('save-sensors');

let floorplanImg = new Image();
let scale = null;
let isCalibrating = false;
let isMeasuring = false;
let isPlacingSensor = false;
let clickCount = 0;
let tempMeasurePoints = [];
let measurementLines = [];
let sensors = [];
let selectedSensor = null;
document.addEventListener('DOMContentLoaded', async () => {
    try {
        const res = await fetch('/sensors');
        const data = await res.json();
        sensors = data.sensors;
        updateSensorList();
        drawFloorplan();
    } catch (e) {
        console.error('Failed to fetch sensors:', e);
    }
});
const uploadInput = document.querySelector('input[type="file"][name="floorplan"]');
uploadInput.addEventListener("change", function(e) {
    const file = e.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function(ev) {
            floorplanImg.src = ev.target.result;
            canvas.width = floorplanImg.width;
            canvas.height = floorplanImg.height;
            canvas.style.display = "block";
            drawFloorplan()
        };
        reader.readAsDataURL(file);
    }
    });
document.getElementById('measureBtn').onclick = () => {
    isCalibrating = true;
    isMeasuring = false;
    isPlacingSensor = false;
    clickCount = 0;
    tempMeasurePoints = [];
    point1.style.display = 'none';
    point2.style.display = 'none';
    drawFloorplan();
};

document.getElementById('newMeasureBtn').onclick = () => {
    if (scale === null) {
        alert("Please define the scale first.");
        return;
    }
    isMeasuring = true;
    isCalibrating = false;
    isPlacingSensor = false;
    clickCount = 0;
    tempMeasurePoints = [];
    drawFloorplan();
};

canvas.addEventListener('click', (e) => {
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    if (isCalibrating) {
        handleCalibrationClick(x, y);
    } else if (isMeasuring) {
        handleMeasurementClick(x, y);
    } else if (isPlacingSensor) {
        addSensor(x, y);
    }
});

document.getElementById('save-sensors').addEventListener('click', () => {
    fetch('/save_sensors', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(sensors)
    }).then(res => alert('Sensors saved.')).catch(err => alert('Failed to save.'));
});

function handleCalibrationClick(x, y) {
    if (clickCount === 0) {
        movePoint(point1, x, y);
        point1.style.display = 'block';
        clickCount++;
    } else if (clickCount === 1) {
        movePoint(point2, x, y);
        point2.style.display = 'block';

        const distance = prompt('Enter the real-world distance between points (in meters):');
        if (!isNaN(distance) && distance > 0) {
            const pxDist = pixelDistance(getPointPos(point1), getPointPos(point2));
            scale = parseFloat(distance) / pxDist;
            scaleText.textContent = scale.toFixed(5) + ' m/pixel';
        }
        
        isCalibrating = false;
    }
    console.log("Clicked calibration", clickCount, x, y);
    drawFloorplan();
}

function handleMeasurementClick(x, y) {
    tempMeasurePoints.push({ x, y });
    if (tempMeasurePoints.length === 2) {
        measurementLines.push({
            p1: { ...tempMeasurePoints[0] },
            p2: { ...tempMeasurePoints[1] },
            id: Date.now()
        });
        tempMeasurePoints = [];
        drawFloorplan();
    }
}

function movePoint(el, x, y) {
    console.log("Moving point to", x, y);
    el.style.left = `${x}px`;
    el.style.top = `${y}px`;
}

function getPointPos(el) {
    return {
        x: parseFloat(el.style.left),
        y: parseFloat(el.style.top)
    };
}

function pixelDistance(p1, p2) {
    return Math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2);
}

function drawFloorplan() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(floorplanImg, 0, 0);

    if (point1.style.display === 'block' && point2.style.display === 'block') {
        drawLine(getPointPos(point1), getPointPos(point2), true);
    }

    measurementLines.forEach(line => drawLine(line.p1, line.p2, false));

    sensors.forEach(sensor => {
        ctx.beginPath();
        ctx.arc(sensor.x, sensor.y, 8, 0, 2 * Math.PI);
        ctx.fillStyle = 'red';
        ctx.fill();
    });

    updateSensorList();
    saveBtn.style.display = sensors.length > 0 ? 'inline-block' : 'none';
}

function drawLine(p1, p2, isCalibration) {
    ctx.beginPath();
    ctx.moveTo(p1.x, p1.y);
    ctx.lineTo(p2.x, p2.y);
    ctx.strokeStyle = isCalibration ? 'red' : 'blue';
    ctx.lineWidth = 2;
    ctx.stroke();

    if (scale !== null) {
        const px = pixelDistance(p1, p2);
        const meters = px * scale;
        ctx.fillStyle = 'black';
        ctx.font = '14px sans-serif';
        ctx.fillText(meters.toFixed(2) + ' m', (p1.x + p2.x) / 2, (p1.y + p2.y) / 2 - 10);
    }
}

function addSensor(x, y) {
    const activeLi = sensorList.querySelector('li.active');

    if (activeLi) {
        const index = [...sensorList.children].indexOf(activeLi);
        if (index >= 0 && index < sensors.length) {
            sensors[index] = { x, y };
            activeLi.textContent = `Sensor ${index + 1}: (${x.toFixed(1)}, ${y.toFixed(1)})`;
        }
    } else {
        // Add new sensor
        sensors.push({ x, y });
    }

    drawFloorplan();
}

function updateSensorList() {
    sensorList.innerHTML = "";
    if (sensors.length === 0) {
        const li = document.createElement("li");
        li.textContent = "No sensors detected";
        sensorList.appendChild(li);
    } else {
        sensors.forEach((sensor, i) => {
            const li = document.createElement("li");
            li.textContent = `Sensor ${i + 1}: (${sensor.x?.toFixed(1) ?? 'N/A'}, ${sensor.y?.toFixed(1) ?? 'N/A'})`;
            li.addEventListener("click", () => {
                Array.from(sensorList.children).forEach(child => child.classList.remove("active"));
                li.classList.add("active");
                selectedSensor = sensor;
                isMeasuring = false;
                isCalibrating = false;
                isPlacingSensor = true;
            });
            sensorList.appendChild(li);
        });
    }
}

[point1, point2].forEach(point => {
    let isDragging = false;
    point.addEventListener('mousedown', () => isDragging = true);
    document.addEventListener('mouseup', () => isDragging = false);
    document.addEventListener('mousemove', (e) => {
        if (!isDragging) return;
        const rect = canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        movePoint(point, x, y);
        drawFloorplan();
    });
});
