bind = "0.0.0.0:5000"
workers = 1             # Keep workers low since GPU memory is shared
timeout = 300           # 5-minute timeout for slow generations
keepalive = 5
accesslog = "-"         # Log access requests to stdout/stderr

