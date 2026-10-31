import os

SCRATCH = ["probe.py", "probe.txt", "probe2.py", "probe2.txt", "probe3.py",
           "check_bg.py", "find_bg.py", "list_files.py", "list_legacy.py",
           "git_status.py", "show_artifacts.py", "scan_api.py",
           "inspect_page.py", "run_smoke.py", "run_pipeline_local.py",
           "_tmp_atest.py", ".pytest_output.txt"]
for f in SCRATCH:
    if os.path.exists(f):
        os.remove(f)
        print("removed", f)

for d in ("__pycache__", ".pytest_cache"):
    if os.path.isdir(d):
        for root, dirs, files in os.walk(d, topdown=False):
            for fn in files:
                os.remove(os.path.join(root, fn))
            for dn in dirs:
                os.rmdir(os.path.join(root, dn))
        os.rmdir(d)
        print("removed tree", d)
print("done")