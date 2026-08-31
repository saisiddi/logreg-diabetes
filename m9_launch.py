"""M9 launch test: really start `streamlit run app.py --server.headless true`,
confirm the server comes up, serves HTTP 200, and logs no traceback.

This is the complement to the `M9` check in `selftest.py`: that one renders the
script and asserts the on-screen numbers, this one proves the actual Streamlit
server process starts cleanly.

Run:  python m9_launch.py
"""

from __future__ import annotations

import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).parent
PORT = 8531
BOOT_TIMEOUT = 120

ERROR_PATTERN = re.compile(
    r"Traceback \(most recent call last\)|ModuleNotFoundError|"
    r"^\s*(Error|Exception|AttributeError|TypeError|ValueError|KeyError|"
    r"NameError|ImportError|FileNotFoundError)\b",
    re.MULTILINE,
)


def main() -> int:
    cmd = [
        sys.executable, "-m", "streamlit", "run", "app.py",
        "--server.headless", "true",
        "--server.port", str(PORT),
        "--browser.gatherUsageStats", "false",
        "--server.fileWatcherType", "none",
    ]
    print("launching:", " ".join(cmd))

    log = ROOT / "m9_launch.log"
    with log.open("w", encoding="utf-8") as fh:
        proc = subprocess.Popen(
            cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT, text=True
        )

        status = None
        health = None
        deadline = time.time() + BOOT_TIMEOUT
        while time.time() < deadline:
            if proc.poll() is not None:
                break
            try:
                with urllib.request.urlopen(
                    f"http://localhost:{PORT}/_stcore/health", timeout=3
                ) as r:
                    health = r.read().decode().strip()
                with urllib.request.urlopen(
                    f"http://localhost:{PORT}/", timeout=5
                ) as r:
                    status = r.status
                break
            except (urllib.error.URLError, ConnectionError, OSError):
                time.sleep(1.0)

        # give the script a moment to finish its first run so errors reach the log
        time.sleep(8)
        alive = proc.poll() is None
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()

    text = log.read_text(encoding="utf-8", errors="replace")
    errors = ERROR_PATTERN.findall(text)

    print(f"\n--- server log ({log.name}) ---")
    print(text.strip() or "(empty)")
    print("--- end log ---\n")

    print(f"health endpoint : {health!r}")
    print(f"HTTP status /   : {status}")
    print(f"process alive   : {alive}")
    print(f"error patterns  : {errors if errors else 'none'}")

    ok = (health == "ok") and (status == 200) and alive and not errors
    print(f"\nM9 launch: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
