import os
import threading
import time
import requests

# =========================
# TẮT HOÀN TOÀN AUTO WAKE KHI CHẠY TRÊN FLY.IO
# =========================
if os.getenv("FLY_APP_NAME"):
    print("✅ Auto wake disabled on Fly.io")

    def start_keep_alive():
        print("✅ start_keep_alive disabled on Fly")

    def mark_activity():
        print("✅ mark_activity disabled on Fly")

    def start_auto_wake_background():
        print("✅ start_auto_wake_background disabled on Fly")

    # ❗ QUAN TRỌNG: DỪNG FILE TẠI ĐÂY, KHÔNG CHẠY BẤT KỲ LOOP NÀO
    raise SystemExit("Auto wake disabled for Fly.io")

# =========================
# PHẦN DƯỚI CHỈ CHẠY Ở LOCAL / SERVER THƯỜNG
# =========================

PING_URL = os.getenv("AUTO_WAKE_URL")
INTERVAL_SECONDS = int(os.getenv("AUTO_WAKE_INTERVAL", "300"))


def _ping_forever():
    if not PING_URL:
        print("[auto_wake] AUTO_WAKE_URL is not set.")
        return

    print(f"[auto_wake] started, ping {PING_URL} every {INTERVAL_SECONDS}s")
    while True:
        try:
            requests.get(PING_URL, timeout=10)
        except Exception as exc:
            print(f"[auto_wake] ping error: {exc}")
        time.sleep(INTERVAL_SECONDS)


def start_auto_wake_background():
    if not PING_URL:
        return
    t = threading.Thread(target=_ping_forever, daemon=True)
    t.start()


def start_keep_alive():
    try:
        print('[auto_wake] start_keep_alive() called (compat)')
    except Exception as exc:
        print(f'[auto_wake] compat start_keep_alive error: {exc}')


def mark_activity():
    try:
        print('[auto_wake] mark_activity() called (compat)')
    except Exception as exc:
        print(f'[auto_wake] compat mark_activity error: {exc}')