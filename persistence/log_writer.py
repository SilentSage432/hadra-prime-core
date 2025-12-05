import json
from datetime import datetime


class LogWriter:

    def __init__(self, filename="prime_runtime.log"):
        self.filename = filename

    def write(self, entry: dict):
        entry["timestamp"] = datetime.utcnow().isoformat()
        with open(self.filename, "a") as f:
            f.write(json.dumps(entry) + "\n")

