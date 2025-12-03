import os
import time
from typing import Optional

CLEANUP_FOLDER = os.getenv("CLEANUP_FOLDER", "saved")
MAX_AGE_HOURS = int(os.getenv("MAX_FILE_AGE_HOURS", "24"))


def cleanup_old_files(folder: Optional[str] = None, max_age_hours: Optional[int] = None) -> int:
    folder = folder or CLEANUP_FOLDER
    max_age_hours = max_age_hours or MAX_AGE_HOURS

    if not os.path.isdir(folder):
        return 0

    now = time.time()
    threshold = now - max_age_hours * 3600
    removed = 0

    for name in os.listdir(folder):
        path = os.path.join(folder, name)
        if not os.path.isfile(path):
            continue
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            continue

        if mtime < threshold:
            try:
                os.remove(path)
                removed += 1
            except OSError:
                pass

    return removed


def run_cleanup_startup() -> None:
    try:
        removed = cleanup_old_files()
        print(f"[auto_cleanup] removed {removed} old files from {CLEANUP_FOLDER}")
    except Exception as exc:
        print(f"[auto_cleanup] error: {exc}")

def start_cleanup_thread():
    # Tuong thích v?i main.py cu
    # G?i d?n d?p ngay khi server start
    try:
        removed = cleanup_old_files()
        print(f'[auto_cleanup] (compat) removed {removed} old files')
    except Exception as exc:
        print(f'[auto_cleanup] (compat) error: {exc}')

def start_cleanup_thread():
    # Hàm tuong thích v?i main.py cu
    # G?i d?n d?p ngay khi server kh?i d?ng
    try:
        removed = cleanup_old_files()
        print(f'[auto_cleanup] (compat) removed {removed} old files')
    except Exception as exc:
        print(f'[auto_cleanup] (compat) error: {exc}')
