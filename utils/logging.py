import csv
from datetime import datetime

def log_crts(query, crts, logfile="crts_log.csv"):
    row = {
        "timestamp": datetime.utcnow().isoformat(),
        "query": query,
        "source_fidelity": crts["source_fidelity"],
        "conflict_reporting": crts["conflict_reporting_rate"],
        "audit_latency": crts["audit_latency_seconds"],
        "guideline_alignment": crts["guideline_alignment"]
    }

    file_exists = os.path.exists(logfile)
    with open(logfile, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
