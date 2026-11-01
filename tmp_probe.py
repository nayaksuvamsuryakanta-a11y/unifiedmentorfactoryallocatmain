from pathlib import Path

from streamlit.testing.v1 import AppTest

at = AppTest.from_file(str(Path("app.py"))).run(timeout=300)
print("selectbox keys:", list(at.selectbox.keys()))
print("sidebar selectbox keys:", list(at.sidebar.selectbox.keys()))
print("multiselect keys:", list(at.multiselect.keys()))
print("slider keys:", list(at.slider.keys()))
sb = at.selectbox["sel_product"]
print("selectbox type:", type(sb), "value:", sb.value)
sb.set_value("Laffy Taffy").run(timeout=180)
print("after set:", at.selectbox["sel_product"].value)
print("subheaders:", [s.value for s in at.subheader])
caps = [c.value for c in at.caption]
print("captions:", caps[:6])
print("download buttons:", list(at.download_button.keys()))
db = at.download_button[list(at.download_button.keys())[0]]
print("download label:", db.value, "file:", db.file_name)
db2 = at.sidebar.selectbox["sel_product"]
print("sidebar selectbox value:", db2.value)
db2.set_value("Nerds").run(timeout=180)
print("after sidebar change:", at.selectbox["sel_product"].value)
print("warnings:", [w.value for w in at.warning])
print("errors:", [e.value for e in at.error])