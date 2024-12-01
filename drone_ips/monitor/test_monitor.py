import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import time

# Hardcoded current_data for testing
current_data = {
    "timestamp": 1731000826.042371,
    "airspeed": 0,
    "armed": True,
    "attitude.pitch": -0.087111294,
    "attitude.roll": -0.013862042,
    "attitude.yaw": 3.0820772647857666,
    "battery.current": 0.24,
    "battery.level": 63,
    "battery.voltage": 15.309,
    "commands.count": 0,
    "commands.next": 0,
    "ekf_ok": False,
    "gps_0.eph": 65,
    "gps_0.epv": 91,
    "gps_0.fix_type": 4,
    "gps_0.satellites_visible": 21,
    "groundspeed": 0.009399953,
    "heading": 176,
    "home_location.alt": 80.918,
    "home_location.lat": 39.1423829,
    "home_location.lon": -76.7304811,
    "is_armable": False,
    "last_heartbeat": 0.024725555000031818,
    "location.global_frame.alt": 80.67,
    "location.global_frame.lat": 39.1423829,
    "location.global_frame.lon": -76.7304817,
    "location.global_relative_frame.alt": 80.67,
    "location.global_relative_frame.lat": 39.1423829,
    "location.global_relative_frame.lon": -76.7304817,
    "location.local_frame.down": -0.009,
    "location.local_frame.east": -0.804621339,
    "location.local_frame.north": -17.14904022,
    "mode.name": "ALTCTL",
    "system_status.state": "ACTIVE",
    "velocity[0]": -0.01,
    "velocity[1]": -0.02,
    "velocity[2]": 0.07,
    "version.autopilot_type": 12,
    "version.major": 1,
    "version.minor": 15,
    "version.patch": 0,
    "version.raw_version": 17760384,
    "version.release": 128,
    "version.vehicle_type": 13
}


def preprocess_vehicle_data(current_data: dict) -> np.array:
    """Preprocess the vehicle data for prediction.

    Parameters
    ----------
    current_data : dict
        The current data from the vehicle.

    Returns
    -------
    np.array
        The preprocessed data ready for prediction.
    """
    # Convert the current_data dictionary to a DataFrame
    df = pd.DataFrame([current_data])

    # Feature Extraction
    features = [ 'gps_0.eph', 'gps_0.epv',
                'gps_0.satellites_visible', 'location.global_frame.lat',
                'location.global_frame.lon', 'location.global_frame.alt',
                'heading' ]

    # Select all columns except those in columns_to_exclude
    df = df[features].copy()

    # Convert necessary columns to numeric values to avoid type errors
    df['location.global_frame.lat'] = pd.to_numeric(df['location.global_frame.lat'], errors='coerce')
    df['location.global_frame.lon'] = pd.to_numeric(df['location.global_frame.lon'], errors='coerce')
    df['location.global_frame.alt'] = pd.to_numeric(df['location.global_frame.alt'], errors='coerce')
    df['heading'] = pd.to_numeric(df['heading'], errors='coerce')
    df['gps_0.eph'] = pd.to_numeric(df['gps_0.eph'], errors='coerce')
    df['gps_0.epv'] = pd.to_numeric(df['gps_0.epv'], errors='coerce')
    df['gps_0.satellites_visible'] = pd.to_numeric(df['gps_0.satellites_visible'], errors='coerce')

    # Fill NaN values which might have been created during conversion to numeric
    df.fillna(0, inplace=True)

    # Feature Engineering
    # Calculate deltas for latitude, longitude, and altitude
    df['delta_lat'] = df['location.global_frame.lat'].diff().fillna(0)
    df['delta_lon'] = df['location.global_frame.lon'].diff().fillna(0)
    df['delta_alt'] = df['location.global_frame.alt'].diff().fillna(0)

    # Calculate Euclidean distance between successive GPS points
    df['distance'] = np.sqrt(df['delta_lat']**2 + df['delta_lon']**2 + df['delta_alt']**2)

    # Load the scaler
    scaler = joblib.load("./model/scaler.pkl")

    # Standardize the data using the loaded scaler
    scaled_data = scaler.transform(df)

    return scaled_data

def load_model(model_path: str):
    """Load the saved ML model.

    Parameters
    ----------
    model_path : str
        The path to the saved model file.

    Returns
    -------
    object
        The loaded model.
    """
    return joblib.load(model_path)

def make_prediction(current_data: dict, model) -> dict:
    """Make a prediction using the ML model.

    Parameters
    ----------
    current_data : dict
        The current data from the vehicle.
    model : object
        The loaded ML model.

    Returns
    -------
    dict
        The prediction result.
    """
    # Preprocess the data
    processed_data = preprocess_vehicle_data(current_data)

    # Make prediction
    prediction = model.predict(processed_data)

    # Return the prediction result in a dictionary
    return {'prediction': int(prediction[0])}

# Main function to test the code
if __name__ == "__main__":
    # Load the model
    model_path = "./model/one_class_svm_model.pkl"
    model = load_model(model_path)

    # Make a prediction
    prediction_result = make_prediction(current_data, model)

    # Print the prediction result
    print("Prediction result:", prediction_result)
