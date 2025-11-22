# ============================================================
# DIRECTORY CONFIG
# ============================================================
from pathlib import Path
import pandas as pd

class DirConfig:
    """
    A class to manage data paths and loading for PM2.5 prediction models.
    """
    def __init__(self, root_dir=Path("."), models_dir_name="models_pipeline_output", meteo_dir_name="meteo_data"):
        self.ROOT = root_dir
        self.METEO_DIR = self.ROOT / meteo_dir_name
        self.PIPE_DIR = self.ROOT / models_dir_name

    def get_meteo_path(self, filename):
        """Constructs the full path to a meteo csv files."""
        path = self.METEO_DIR / filename
        return path
    
    def load_meteo_data(self):
        """Loads a CSV meteorological data files."""
        meteo_path = self.METEO_DIR
        if meteo_path.exists():
            meteo_csv_files = {
                "pm25": self.get_meteo_path("Total-Surface-Mass-Concentration-PM2.5.csv"),
                "dew_temp": self.get_meteo_path("2-meter dew point temperature.csv"),
                "pbl": self.get_meteo_path("Planetary boundary layer height.csv"),
                "surface_air_temp": self.get_meteo_path("Surface air temperature.csv"),
                "surface_pressure": self.get_meteo_path("Surface pressure.csv"),
                "surface_skin_temp": self.get_meteo_path("Surface skin temperature.csv"),
                "surface_wind_temp": self.get_meteo_path("Surface wind speed.csv"),
                "surface_precipitation": self.get_meteo_path("Total surface precipitation.csv")
            }
            return meteo_csv_files
        else:
            raise FileNotFoundError(f"Meteorological data files not found. Check the directory: {meteo_path}")

    def get_models_pipe_path(self):
        """Constructs the full path to a model's pipeline data dir."""
        pipe_dir = self.PIPE_DIR 
        pipe_dir.mkdir(parents=True, exist_ok=True)
        path = pipe_dir
        return path
