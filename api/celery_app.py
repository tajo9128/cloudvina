
import os
from celery import Celery

# Get Redis Config
# Use UPSTASH_REDIS_URL if set, otherwise fallback to local REDIS_URL or localhost
redis_url = os.getenv("UPSTASH_REDIS_URL", os.getenv("REDIS_URL", "redis://localhost:6379/0"))

# Require SSL for Upstash (rediss://)
if "upstash" in redis_url and not redis_url.startswith("rediss://"):
    redis_url = redis_url.replace("redis://", "rediss://")

celery_app = Celery(
    "worker",
    broker=redis_url,
    backend=redis_url
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    # Worker Settings
    worker_concurrency=1,  # Since Colab has 1 GPU, usually 1 worker process is best
    worker_prefetch_multiplier=1, # One task at a time
)

# Optional: Auto-discover tasks if they are in other modules
# celery_app.autodiscover_tasks(['api.services']) 
