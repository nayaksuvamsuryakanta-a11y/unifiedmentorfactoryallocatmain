import os

for dirpath, dirnames, filenames in sorted(os.walk("legacy")):
    dirnames.sort()
    for fn in sorted(filenames):
        p = os.path.join(dirpath, fn)
        print("%10d  %s" % (os.path.getsize(p), os.path.relpath(p)))