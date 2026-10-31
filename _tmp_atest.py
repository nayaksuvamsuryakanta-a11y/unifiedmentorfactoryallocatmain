from streamlit.testing.v1 import AppTest

at = AppTest.from_file("app.py")
at.run(timeout=300)
exc = at.exception
exc_list = exc if isinstance(exc, list) else ([exc] if exc else [])
print("exception count:", len(exc_list))
for e in exc_list:
    print("  exception:", repr(e)[:400])
print("title:", at.title[0].value if at.title else None)
print("dataframes:", len(at.dataframe), " | metrics:", len(at.metric),
      " | sidebars:", len(at.sidebar))
try:
    print("metric values:", {k: v.value for k, v in at.metric.items()})
except Exception as e:
    print("metric err:", type(e).__name__, e)
try:
    at.sidebar.selectbox[0].set_value("Laffy Taffy").run(timeout=180)
    print("after selectbox change, dataframes:", len(at.dataframe))
except Exception as e:
    print("selectbox err:", type(e).__name__, str(e)[:400])
try:
    at.sidebar.slider[0].set_value(0.75).run(timeout=120)
    print("after slider change to 0.75, dataframes:", len(at.dataframe))
except Exception as e:
    print("slider err:", type(e).__name__, str(e)[:400])
try:
    at.sidebar.multiselect[0].set_value(["Atlantic"]).run(timeout=120)
    print("after multiselect change, dataframes:", len(at.dataframe))
except Exception as e:
    print("multiselect err:", type(e).__name__, str(e)[:400])