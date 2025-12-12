
import re

with open('web/src/pages/3DViewer.jsx', 'r', encoding='utf-8') as f:
    content = f.read()

open_divs = len(re.findall(r'<div\b', content))
close_divs = len(re.findall(r'</div>', content))
dirty_divs = len(re.findall(r'</div >', content))

print(f"Open: {open_divs}")
print(f"Close: {close_divs}")
print(f"Dirty: {dirty_divs}")
print(f"Total Close: {close_divs + dirty_divs}")
print(f"Balance: {open_divs - (close_divs + dirty_divs)}")
