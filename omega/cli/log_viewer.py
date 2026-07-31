from __future__ import annotations

import time
from pathlib import Path
from omega.cli.logging_config import get_log_file

def _read_last_lines(path: Path, line_count: int) -> str:
    with path.open("r", encoding="utf-8", errors="replace") as log_file:
        return "".join(log_file.readlines()[-line_count:])

def show_logs(line_count: int = 100, follow: bool = False) -> int:
    if line_count < 1:
        raise ValueError("line_count must be at least 1")

    path = get_log_file()
    if not path.exists():
        print(f"No omega log file exists yet at: {path}")
        return 0

    print(_read_last_lines(path, line_count), end="")
    if not follow:
        return 0

    try:
        with path.open("r", encoding="utf-8", errors="replace") as log_file:
            log_file.seek(0, 2)
            while True:
                line = log_file.readline()
                if line:
                    print(line, end="", flush=True)
                else:
                    time.sleep(0.25)
    except KeyboardInterrupt:
        print()
        return 0