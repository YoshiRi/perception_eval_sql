"""
Start the RQ worker. Run from evaluation_dashboard_app root so worker and lib are importable.
Usage: python -m worker.run_worker
Requires: REDIS_URL, DATABASE_URL, and USE_TASK_QUEUE=true for full operation.
"""

import os
import sys
import faulthandler

_APP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _APP_ROOT not in sys.path:
    sys.path.insert(0, _APP_ROOT)
os.chdir(_APP_ROOT)
faulthandler.enable(all_threads=True)

def main():
    from rq import Worker
    from rq import Queue
    from redis import Redis

    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    conn = Redis.from_url(redis_url)
    queue_name = os.environ.get("RQ_QUEUE", "default")
    queue = Queue(queue_name, connection=conn)

    worker = Worker([queue], connection=conn)
    worker.work(with_scheduler=False)


if __name__ == "__main__":
    main()
