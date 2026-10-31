import os
import re
import sys

home = os.path.expanduser("~")
needle = "shell-bg_c6399b8d"
for root, dirs, files in os.walk(os.path.join(home, ".qwen", "tmp")):
    for f in files:
        if needle in f:
            print(os.path.join(root, f))