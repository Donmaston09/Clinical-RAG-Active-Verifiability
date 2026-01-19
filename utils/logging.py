
# modules/logging.py
"""
CRTS logging utilities
- CSV logger: stable schema matching compute_crts()
- JSONL logger: rolling line-delimited JSON for dashboards/ingest
- Convenience: log_crts_both() writes to CSV + JSONL in one call
"""
from typing import Dict, Optional
from datetime import datetime
import csv
import json
import os

FIELDS = [
    "timestamp", "query", "sf", "crr", "ar", "ga", "crts", "L",
    "alpha", "beta", "gamma", "delta"
]


def _row_from_crts(query: str, crts: Dict) -> Dict:
    """Create a flat row dict from the compute_crts() output for logging."""
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "query": query,
        "sf": crts.get("sf"),
        "crr": crts.get("crr"),
        "ar": crts.get("ar"),
        "ga": crts.get("ga"),
        "crts": crts.get("crts"),
        "L": crts.get("L"),
        "alpha": (crts.get("weights", {}) or {}).get("alpha"),
        "beta": (crts.get("weights", {}) or {}).get("beta"),
        "gamma": (crts.get("weights", {}) or {}).get("gamma"),
        "delta": (crts.get("weights", {}) or {}).get("delta"),
    }


def log_crts(query: str, crts: Dict, logfile: str = "crts_log.csv") -> None:
    """Append a CSV record for the current CRTS."""
    row = _row_from_crts(query, crts)
    file_exists = os.path.exists(logfile)
    with open(logfile, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def log_crts_jsonl(
    query: str,
    crts: Dict,
    jsonl_path: str = "crts_log.jsonl",
    extra: Optional[Dict] = None,
) -> None:
    """Append a JSON line for the current CRTS (rolling JSONL)."""
    data = _row_from_crts(query, crts)
    if extra:
        data.update(extra)
    # Defensive dump (no f-strings, explicit newline)
    line = json.dumps(data, ensure_ascii=False)
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def log_crts_both(
    query: str,
    crts: Dict,
    csv_path: str = "crts_log.csv",
    jsonl_path: str = "crts_log.jsonl",
    extra: Optional[Dict] = None,
) -> None:
    """Write both CSV and JSONL entries in one call."""
    log_crts(query, crts, logfile=csv_path)
    log_crts_jsonl(query, crts, jsonl_path=jsonl_path, extra=extra)
