import csv
import os
from datetime import datetime

def log_crts(query, crts):
    log_file = "logs/crts_performance.csv"

    # Ensure the logs directory exists
    os.makedirs("logs", exist_ok=True)

    file_exists = os.path.isfile(log_file)

    with open(log_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "timestamp", "query", "sf", "crr", "ar", "ga", "L"
        ])

        if not file_exists:
            writer.writeheader()

        writer.writerow({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "query": query,
            "sf": crts.get("sf", 0),
            "crr": crts.get("crr", 0),
            "ar": crts.get("ar", 0),
            "ga": crts.get("ga", 0),
            "L": crts.get("L", 0)
        })
