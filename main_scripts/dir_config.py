# ============================================================
# DIRECTORY CONFIG
# ============================================================
from pathlib import Path

class DirConfig:
    def __init__(self, 
                 root_dir=Path("."), 
                 models_dir_name="models_pipeline_output", 
                 meteo_dir_name="meteo_data",
                 plots_dir_name="plot_figures"
                 ):
        self.root = root_dir
        self.meteo_dir = self.root / meteo_dir_name
        self.models_dir = self.root / models_dir_name
        self.plots_dir = self.root / plots_dir_name

    def get_meteo_path(self, filename):
        path = self.meteo_dir / filename
        return path
    
    def load_meteo_data(self):
        meteo_path = self.meteo_dir
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
        
    def plots_dir_path(self):
        plots_dir = self.plots_dir
        plots_dir.mkdir(parents=True, exist_ok=True)
        path = plots_dir
        return path
            
    def models_dir_path(self):
        models_dir = self.models_dir 
        models_dir.mkdir(parents=True, exist_ok=True)
        path = models_dir
        return path
