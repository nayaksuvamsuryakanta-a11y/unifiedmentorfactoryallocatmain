import os

root = os.getcwd()
files = []
for dirpath, dirnames, filenames in os.walk(root):
    # skip hidden env and git/cache dirs
    dn = [d for d in dirnames if d not in (".venv", ".git", "__pycache__", ".pytest_cache", "legacy")]
    dirnames[:] = dn
    for fn in filenames:
        if fn.endswith((".pyc",)):
            continue
        p = os.path.join(dirpath, fn)
        rel = os.path.relpath(p, root)
        size = os.path.getsize(p)
        files.append((rel, size))
files.sort()
for rel, size in files:
    print("%10d  %s" % (size, rel))
print("\nTOTAL FILES:", len(files))