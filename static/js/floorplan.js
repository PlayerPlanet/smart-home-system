document.addEventListener("DOMContentLoaded", async function() {
    // Fetch sensors and populate list
    const sensorList = document.getElementById("sensor-list");
    let sensors = [];
    let selectedSensor = null;
    let sensorPositions = {};

    async function fetchSensors() {
        try {
            const response = await fetch("/sensors");
            const data = await response.json();
            sensors = data.sensors || [];
            sensorList.innerHTML = "";
            if (sensors.length === 0) {
                const li = document.createElement("li");
                li.textContent = "No sensors detected";
                sensorList.appendChild(li);
            } else {
                sensors.forEach(sensor => {
                    const li = document.createElement("li");
                    li.textContent = sensor;
                    li.addEventListener("click", () => {
                        Array.from(sensorList.children).forEach(child => child.classList.remove("active"));
                        li.classList.add("active");
                        selectedSensor = sensor;
                        showCrosshair();
                    });
                    sensorList.appendChild(li);
                });
            }
        } catch (err) {
            sensorList.innerHTML = "<li>Failed to fetch sensors</li>";
        }
    }

    await fetchSensors();

    // Handle floorplan image preview on file upload
    const uploadInput = document.querySelector('input[type="file"][name="floorplan"]');
    const floorplanImg = document.getElementById("floorplan-img");
    const floorplanContainer = document.getElementById("floorplan-container");

    uploadInput.addEventListener("change", function(e) {
        const file = e.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = function(ev) {
                floorplanImg.src = ev.target.result;
                floorplanImg.style.display = "block";
            };
            reader.readAsDataURL(file);
        }
    });

    // Crosshair and sensor placement
    let crosshair = null;
    function showCrosshair() {
        if (!crosshair) {
            crosshair = document.createElement("div");
            crosshair.className = "crosshair";
            floorplanContainer.appendChild(crosshair);
        }
        crosshair.style.display = selectedSensor ? "block" : "none";
    }

    floorplanImg.addEventListener("click", function(e) {
        if (!selectedSensor) return;
        const rect = floorplanImg.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        // Place dot
        let dot = document.querySelector(`.sensor-dot[data-sensor="${selectedSensor}"]`);
        if (!dot) {
            dot = document.createElement("div");
            dot.className = "sensor-dot";
            dot.dataset.sensor = selectedSensor;
            floorplanContainer.appendChild(dot);
        }
        dot.style.left = `${x}px`;
        dot.style.top = `${y}px`;

        // Move crosshair to this position
        if (crosshair) {
            crosshair.style.left = `${x}px`;
            crosshair.style.top = `${y}px`;
        }

        // Save position
        sensorPositions[selectedSensor] = { x, y };
        document.getElementById("save-sensors").style.display = "block";
    });

    // Save sensor positions and scale
    document.getElementById("save-sensors").addEventListener("click", async function() {
        const scalePixels = parseFloat(document.getElementById("scale-pixels").value);
        const scaleMeters = parseFloat(document.getElementById("scale-meters").value);
        if (!scalePixels || !scaleMeters) {
            alert("Please define the scale.");
            return;
        }
        if (Object.keys(sensorPositions).length === 0) {
            alert("Please place at least one sensor.");
            return;
        }
        // Prepare data
        const data = {
            sensor_positions: sensorPositions,
            scale: { pixels: scalePixels, meters: scaleMeters }
        };
        const formData = new FormData();
        formData.append("image",floorplanImg);
        formData.append("metadata", new Blob(
        [JSON.stringify(data)],
        { type: "application/json" }
        ));
        // Send POST to /floorplan
        const resp = await fetch("/floor", {
            method: "POST",
            body: formData
        });
        if (resp.ok) {
            alert("Floorplan and sensor positions saved!");
        } else {
            alert("Failed to save floorplan data.");
        }
    });

    // Form submit: upload floorplan image to backend
    document.getElementById("upload-form").addEventListener("submit", async function(e) {
        e.preventDefault();
        const formData = new FormData(this);
        const resp = await fetch("/floor", {
            method: "POST",
            body: formData
        });
        if (resp.ok) {
            const data = await resp.json();
            if (data.floorplan_url) {
                floorplanImg.src = data.floorplan_url;
            }
        } else {
            alert("Failed to upload floorplan image.");
        }
    });
});