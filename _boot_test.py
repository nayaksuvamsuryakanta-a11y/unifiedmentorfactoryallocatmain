"""Focused AppTest probe, run from the project root."""
import sys
import time

from streamlit.testing.v1 import AppTest

t0 = time.time()
print("[probe] building AppTest...", flush=True)
at = AppTest.from_file("app.py")
print("[probe] built in %.1fs, running with timeout=120..." % (time.time() - t0), flush=True)
at.run(timeout=120)
print("[probe] run took %.1fs" % (time.time() - t0), flush=True)
exc = at.exception
print("[probe] exception:", exc, flush=True)
if not exc:
    print("[probe] pages:", sorted(at.page_names), flush=True)
    side = at.sidebar
    print("[probe] sidebar selectbox:", len(side.selectbox), flush=True)
    print("[probe] sidebar multiselect:", len(side.multiselect), flush=True)
    print("[probe] sidebar slider:", len(side.slider), flush=True)
    print("[probe] dataframes:", len(at.dataframe), flush=True)
    if at.dataframe:
        print("[probe] first dataframe shape:", at.dataframe[0].value.shape, flush=True)