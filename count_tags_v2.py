
import re

def remove_comments(text):
    # Remove // comments
    text = re.sub(r'//.*', '', text)
    # Remove /* */ comments
    text = re.sub(r'/\*[\s\S]*?\*/', '', text)
    return text

with open('web/src/pages/3DViewer.jsx', 'r', encoding='utf-8') as f:
    content = f.read()

clean_content = remove_comments(content)

open_divs = len(re.findall(r'<div\b', clean_content))
close_divs = len(re.findall(r'</div>', clean_content))

print(f"Clean Open: {open_divs}")
print(f"Clean Close: {close_divs}")
print(f"Balance: {open_divs - close_divs}")

# Check for specific blocks
lines = clean_content.splitlines()
for i, line in enumerate(lines):
    if '<div' in line or '</div>' in line:
        pass # Just scanning
