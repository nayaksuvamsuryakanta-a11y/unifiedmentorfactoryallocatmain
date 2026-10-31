import subprocess
import sys

PY = r".venv\Scripts\python.exe"
cmd = [PY, "-m", "pytest", "scripts/smoke_test.py", "-v", "-p", "no:randomly"]
print("RUN:", " ".join(cmd), flush=True)
r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
print("EXIT:", r.returncode)
print(r.stdout)
print(r.stderr)
with open("pytest_run.log", "w") as f:
    f.write("EXIT %d\n\n" % r.returncode)
    f.write(r.stdout)
    f.write("\n--- STDERR ---\n")
    f.write(r.stderr)
print("WROTE pytest_run.log")
sys.exit(r.returncode)