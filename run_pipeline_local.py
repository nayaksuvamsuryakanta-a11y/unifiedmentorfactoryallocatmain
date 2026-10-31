import subprocess
import sys

PY = r".venv\Scripts\python.exe"
r = subprocess.run([PY, "run_pipeline.py"], capture_output=True, text=True, timeout=900)
print("EXIT:", r.returncode)
print(r.stdout)
print(r.stderr)
with open("pipeline_run.log", "w") as f:
    f.write("EXIT %d\n\n" % r.returncode)
    f.write(r.stdout)
    f.write("\n--- STDERR ---\n")
    f.write(r.stderr)
sys.exit(r.returncode)