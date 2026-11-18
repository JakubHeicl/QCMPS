from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
 
class Logger:
    def __init__(
        self, 
        log_file: Path | str = "experiment.log", 
        value_file: Path | str | None = None,
        continue_logging: bool = False
    ):
        self.log_file = Path(log_file)
        self.value_file = Path(value_file) if value_file is not None else None
        self.continue_logging = continue_logging
        
        self.funev = 0
        
        if not self.continue_logging:
            with open(self.log_file, "w", encoding="utf-8") as f:
                f.write("")
                
            if self.value_file:
                with open(self.value_file, "w", encoding="utf-8") as f:
                    f.write("funev,metric,value\n")

        if self.continue_logging and not self.log_file.exists():
            raise FileNotFoundError(f"Log file {self.log_file} does not exist. Set continue_logging to False to create a new log file.")
        
        if self.continue_logging and self.value_file and not self.value_file.exists():
            raise FileNotFoundError(f"Value file {self.value_file} does not exist. Set continue_logging to False to create a new value file.")

        if self.continue_logging and self.value_file is not None:
            with open(self.value_file, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]

            if lines:
                try:
                    self.funev = int(lines[-1].split(",", maxsplit=1)[0])
                except (ValueError, IndexError):
                    pass

    def log(self, message: str):
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(message + "\n")
        print(message)
        
    def log_value(self, metric: str, value: float, delta_funev: int = 1):
            if delta_funev < 0:
                raise ValueError("delta_funev must be non-negative.")

            self.funev += delta_funev

            if self.value_file is None:
                return

            with open(self.value_file, "a", encoding="utf-8") as f:
                f.write(f"{self.funev},{metric},{value:.16g}\n")
                
    def plot_values(self, metric: str, log_scale: bool = False):
        if self.value_file is None:
            raise ValueError("Value file is not specified.")

        df = pd.read_csv(self.value_file)
        df_metric = df[df["metric"] == metric]

        plt.figure(figsize=(10, 6))
        plt.plot(df_metric["funev"], df_metric["value"], marker='o')
        plt.title(f"{metric}")
        plt.xlabel("Function Evaluations")
        plt.ylabel(metric)
        if log_scale:
            plt.yscale("log")
        plt.grid()
        plt.show()
        
    def plot_error(self, metric: str, reference_value: float, log_scale: bool = False):
        if self.value_file is None:
            raise ValueError("Value file is not specified.")

        df = pd.read_csv(self.value_file)
        df_metric = df[df["metric"] == metric]
        df_metric["error"] = (df_metric["value"] - reference_value).abs()

        plt.figure(figsize=(10, 6))
        plt.plot(df_metric["funev"], df_metric["error"], linewidth=1)
        plt.title(f"Error in {metric} (Reference: {reference_value})")
        plt.xlabel("Function Evaluations")
        plt.ylabel("Absolute Error")
        if log_scale:
            plt.yscale("log")
        plt.grid()
        plt.show()