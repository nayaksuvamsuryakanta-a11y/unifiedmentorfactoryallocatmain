import subprocess

r = subprocess.run(["git", "status", "--short"], capture_output=True, text=True)
print("git status exit:", r.returncode)
print(r.stdout)
if r.stderr:
    print("STDERR:", r.stderr[:500])

r2 = subprocess.run(["git", "log", "-3", "--oneline"], capture_output=True, text=True)
print("recent commits:")
print(r2.stdout)