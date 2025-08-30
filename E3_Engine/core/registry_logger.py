"""
registry_logger.py
------------------
Standardized event logging into a CSV registry.
"""

import csv
from pathlib import Path
from typing import Any, Dict, List


def log_event(
    event: Dict[str, Any],
    csv_path: str,
    fieldnames: List[str] = None
) -> None:
    """
    Append a single event (row) to a CSV registry.

    If the file does not exist, it will be created and a header row written.

    Args:
        event:     A mapping of column names to values for this event.
        csv_path:  Path to the CSV file (will be created if missing).
        fieldnames:
            Optional list of column names to use (in order).
            If None, uses the keys of `event` in arbitrary order.
    """
    path = Path(csv_path)
    write_header = not path.exists()

    # Determine columns
    if fieldnames is None:
        fieldnames = list(event.keys())

    # Ensure parent directories exist
    path.parent.mkdir(parents=True, exist_ok=True)

    # Open and write
    with open(path, mode="a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(event)
