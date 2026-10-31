import inspect
from streamlit.commands.navigation import Page
print("Page signature:", inspect.signature(Page.__init__))
print("Page docstring excerpt:", (Page.__doc__ or "")[:400])
import functools
p = functools.partial(lambda x: x, 1)
print("partial has __name__:", hasattr(p, "__name__"), getattr(p, "__name__", "N/A"))
print("partial has __qualname__:", getattr(p, "__qualname__", "N/A"))
print("partial __module__:", getattr(p, "__module__", "N/A"))