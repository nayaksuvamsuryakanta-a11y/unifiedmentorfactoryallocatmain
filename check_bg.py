import os
import time

for f in ("pytest_run.log", "pipeline_run.log"):
    if os.path.exists(f):
        print("%-18s %10d bytes  mtime %s" % (f, os.path.getsize(f),
              time.strftime("%H:%M:%S", time.localtime(os.path.getmtime(f)))))
    else:
        print("%-18s not yet written" % f)