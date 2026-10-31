import os, glob, sys, importlib.util

out = os.path.join(os.getcwd(), "probe.txt")
with open(out, "w") as f:
    f.write("cwd: %s\n" % os.getcwd())
    f.write("sys.executable: %s\n" % sys.executable)
    f.write("venv exists: %s\n" % os.path.isdir("venv"))
    f.write("venv python: %s\n" % os.path.isfile(r"venv\Scripts\python.exe"))
    for p in sorted(glob.glob("venv/Scripts/*.exe")):
        f.write("  %s\n" % p)
    f.write("pytest importable: %s\n" % (importlib.util.find_spec("pytest") is not None))
    f.write("streamlit importable: %s\n" % (importlib.util.find_spec("streamlit") is not None))
print("WROTE", out, os.path.exists(out))