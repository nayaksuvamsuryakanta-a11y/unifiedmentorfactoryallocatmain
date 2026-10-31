import importlib.util, subprocess, sys
PY = r".venv\Scripts\python.exe"
print("python:", PY)
for name in ("streamlit", "pytest", "pandas", "numpy", "sklearn", "plotly", "joblib", "scipy", "matplotlib", "seaborn", "tenacity", "PIL"):
    print("  %-12s %s" % (name, subprocess.run([PY, "-c", "import importlib.util,sys; "
          "sys.exit(0 if importlib.util.find_spec('%s') else 1)" % name]).returncode == 0))
print("version:", subprocess.run([PY, "-V"], capture_output=True, text=True).stdout.strip())
print("streamlit version:", subprocess.run([PY, "-m", "streamlit", "--version"],
      capture_output=True, text=True).stdout.strip()[:60])