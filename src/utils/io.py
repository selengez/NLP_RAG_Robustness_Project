# Simple helper for saving data to a JSON file.

import json
from pathlib import Path


def save_json(data, path):
    """Save data (dict or list) to a JSON file, creating directories if needed."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
