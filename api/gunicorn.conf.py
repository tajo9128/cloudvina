import os
import sys

# Gunicorn Configuration to force logging to stdout/stderr
# This file is automatically loaded by gunicorn if present in the working directory

# Server Socket
bind = f"0.0.0.0:{os.getenv('PORT', '10000')}"
workers = 4
worker_class = 'uvicorn.workers.UvicornWorker'
timeout = 120

# Logging
accesslog = '-'       # Log to stdout
errorlog = '-'        # Log to stderr
loglevel = 'info'
capture_output = True # Capture stdout/stderr from workers
enable_stdio_inheritance = True

# Formatting
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s"'

print("Gunicorn config loaded: Logging to stdout/stderr enabled", file=sys.stdout, flush=True)
