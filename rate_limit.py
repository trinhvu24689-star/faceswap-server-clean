import time
from fastapi import HTTPException, status

_REQUEST_LOG = {}
MAX_PER_MINUTE = 20
WINDOW_SECONDS = 60


def check_rate_limit(client_id: str) -> None:
    if not client_id:
        client_id = "unknown"

    now = time.time()
    window_start = now - WINDOW_SECONDS

    history = _REQUEST_LOG.get(client_id, [])
    history = [t for t in history if t >= window_start]
    history.append(now)
    _REQUEST_LOG[client_id] = history

    if len(history) > MAX_PER_MINUTE:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many requests. Please slow down.",
        )
