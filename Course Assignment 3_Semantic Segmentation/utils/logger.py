# utils/logger.py
import csv
import os


class CSVLogger:
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.header_written = False

    def log(self, **kwargs):
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        write_header = not os.path.exists(self.csv_path) or not self.header_written

        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=sorted(kwargs.keys()))
            if write_header:
                writer.writeheader()
                self.header_written = True
            writer.writerow(kwargs)

