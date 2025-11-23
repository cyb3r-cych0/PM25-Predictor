from plot_scripts.actual_vs_pred_trends_plots import ActualVsPredTrendsPlotter
from plot_scripts.pred_vs_obs_plots import PredVsObsPlotter
from plot_scripts.bias_trend_timeseries import BiasTrendTimeseriesPlotter

class RunMain:
    def __init__(self):
        self.actual_vs_pred = ActualVsPredTrendsPlotter()
        self.pred_vs_obs = PredVsObsPlotter()
        self.bias_trend_tseries = BiasTrendTimeseriesPlotter()
    
    def run(self):  

        print("Plotting Actual vs PredictedTtrend Plots..")
        self.actual_vs_pred.run()

        print("Plotting Predicted vs Observations Plots..")
        self.pred_vs_obs.run()

        print("Plotting Bias (residual) Trend Timeseries")
        self.bias_trend_tseries.run()

if __name__ == "__main__":
    run = RunMain()
    run.run()