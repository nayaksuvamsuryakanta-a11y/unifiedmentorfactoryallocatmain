import os, glob, importlib.util

lines = []
lines.append("cwd: %s" % os.getcwd())
lines.append("sys.executable: %s" % os.getenv("PYTHONEXECUTABLE", "-"))
for name in ("streamlit", "pytest", "pandas", "sklearn"):
    lines.append("module %s: %s" % (name, importlib.util.find_spec(name) is not None))

# look for venvs nearby
for root in (os.getcwd(), os.path.dirname(os.getcwd()), os.path.expanduser("~")):
    if not root:
        continue
    for pat in ("venv", ".venv", "env"):
        p = os.path.join(root, pat)
        py = os.path.join(p, "Scripts", "python.exe")
        if os.path.isfile(py):
            lines.append("FOUND venv python: %s" % py)
lines.append("streamlit on PATH (where): ")
for line in os.popen("where streamlit 2>nul").read().splitlines():
    lines.append("  " + line)
lines.append("pytest on PATH (where): ")
for line in os.popen("where pytest 2>nul").read().splitlines():
    lines.append("  " + line)
lines.append("python -m streamlit --version: ")
for line in os.popen("python -m streamlit --version 2>nul").read().splitlines()[:3]:
    lines.append("  " + line)
with open("probe2.txt", "w") as f:
    f.write("\n".join(lines) + "\n")
print("done")