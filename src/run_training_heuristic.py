"""
Smart Home IoT System - Heuristic Model Training Script

This script performs training and optimization of heuristic device ranking
parameters using historical sensor data and user ratings. It demonstrates
the complete machine learning pipeline for improving device importance
prediction accuracy.

The training process:
1. Loads all historical sensor data from the database
2. Parses sensor measurements into structured format
3. Runs parameter optimization using user ratings as ground truth
4. Outputs optimized parameters for production use

Usage:
    python -m src.run_training_heuristic

Requirements:
    - Populated sensor database (sensor_data.db)
    - User ratings file (src/model_data/ratings_log.json)
    - Sufficient training data for stable optimization

Output:
    Prints optimization results including:
    - Best parameter values (rssi_ref, rssi_min, N_thresh, half_life)
    - Optimization convergence status
    - Final objective function value (MSE)

Author: Smart Home IoT Team
Version: 1.0.0
"""

from .app import _fetch_db_data, _parse_sensor_row
from .analyze_data import train_heuristic_params

if __name__ == "__main__":
    """
    Main training execution script.
    
    Executes the complete heuristic model training pipeline:
    1. Fetches all sensor data from SQLite database
    2. Parses JSON strings into structured SensorData objects  
    3. Runs parameter optimization with 5 sensors as system maximum
    4. Prints optimization results for analysis
    
    Example Output:
        ```
        fun: 0.0234
        success: True
        x: array([-62.3, -95.7, 8.2, 28500.0])
        message: 'CONVERGENCE: REL_REDUCTION_OF_F <= FACTR*EPSMCH'
        ```
        
    Where x contains optimized [rssi_ref, rssi_min, N_thresh, half_life] values.
    
    Note:
        The S_max parameter (5) should match your actual sensor count for
        accurate normalization during optimization.
    """
    all_data = _fetch_db_data()
    parsed_data = [_parse_sensor_row(row) for row in all_data]
    res = train_heuristic_params(parsed_data, 5)
    print(res)