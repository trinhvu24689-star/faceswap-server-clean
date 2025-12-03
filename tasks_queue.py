from queue import Queue
from threading import Thread
from typing import Callable, Any


class TaskQueue:
    def __init__(self):
        self._queue: "Queue[Callable[[], Any]]" = Queue()
        self._worker = Thread(target=self._run, daemon=True)
        self._worker.start()

    def _run(self):
        while True:
            try:
                job = self._queue.get()
                job()
            except Exception as exc:
                print(f"[tasks_queue] job error: {exc}")

    def submit(self, fn: Callable[[], Any]) -> None:
        self._queue.put(fn)


task_queue = TaskQueue()
