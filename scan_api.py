import ast
import inspect
import re
import streamlit

src = open("app.py", encoding="utf-8").read()
tree = ast.parse(src)

top = {n for n in dir(streamlit) if not n.startswith("_")} | {"column_config", "errors"}
nested_ok = True
for node in ast.walk(tree):
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Attribute):
        if ast.unparse(node.value).startswith("st."):
            first = node.value.attr
            second = node.attr
            mod = getattr(streamlit, first, None)
            if mod is None:
                print("MISSING: st.%s (no such submodule/attribute)" % first)
                nested_ok = False
            elif not hasattr(mod, second):
                print("MISSING: st.%s.%s" % (first, second))
                nested_ok = False
if nested_ok:
    print("all nested st.<sub>.<name> accesses resolve")

# catch probable typos of known names near the typo'd forms
known = {"ColumnConfig", "Column_Conf", "column_config"}
for m in re.finditer(r"st\.([A-Za-z_]\w*)\.([A-Za-z_]\w*)", src):
    pass
print("module members of st.column_config:", sorted(n for n in dir(streamlit.column_config)
                                                    if not n.startswith("_")))
print("module members of st.errors:", sorted(n for n in dir(streamlit.errors)
                                             if not n.startswith("_")))