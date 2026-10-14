 
class Logger:
    def __init__(self, log_file: str = "optimization_log.txt"):
        self.logs = []
        self.log_file = log_file
        
        with open(self.log_file, "w") as f:
            f.write("")

    def log(self, message: str):
        
        self.logs.append(message)
        with open(self.log_file, "a") as f:
            f.write(message + "\n")
        print(message)  # Print to console for immediate feedback