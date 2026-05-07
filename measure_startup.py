import time
import urllib.request
import urllib.error

start = time.monotonic()
url = "http://localhost:8000/health"

print(f"Waiting for vLLM at {url}...")

while True:
    try:
        resp = urllib.request.urlopen(url, timeout=2)
        if resp.status == 200:
            elapsed = time.monotonic() - start
            print(f"Startup time: {elapsed:.1f}s")
            break
    except (urllib.error.URLError, OSError):
        pass
    time.sleep(0.2)
