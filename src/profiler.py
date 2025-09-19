import time
import torch
import csv
import os

class Profiler:
    def __enter__(self):
        self.start_time = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        self.total_time = self.end_time - self.start_time
        if torch.cuda.is_available():
            self.peak_memory_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        else:
            self.peak_memory_gb = 0

class DetailedLogger:
    def __init__(self, run_name, log_dir="results/details"):
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, f"{run_name}_metrics.csv")
        self.file = open(self.log_path, 'w', newline='')
        self.writer = None
        self.headers_written = False

    def log(self, data_dict):
        if not self.headers_written:
            self.writer = csv.DictWriter(self.file, fieldnames=data_dict.keys())
            self.writer.writeheader()
            self.headers_written = True
        self.writer.writerow(data_dict)

    def close(self):
        self.file.close()
