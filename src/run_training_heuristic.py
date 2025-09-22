from .app import _fetch_db_data, _parse_sensor_row
from .analyze_data import train_heuristic_params

if __name__ == "__main__":
    all_data = _fetch_db_data()
    parsed_data = [_parse_sensor_row(row) for row in all_data]
    res = train_heuristic_params(parsed_data, 5)
    print(res)