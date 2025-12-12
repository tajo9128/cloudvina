
import re

def remove_comments(text):
    text = re.sub(r'//.*', '', text)
    text = re.sub(r'/\*[\s\S]*?\*/', '', text)
    return text

with open('web/src/pages/3DViewer.jsx', 'r', encoding='utf-8') as f:
    content = f.read()

# Naive JSX parser
# We only care about <div> and </div> for now since the error is about div
# But we should track all tags ideally? No, just div to find the mismatch in divs.
# The error is specific to div.

clean_content = remove_comments(content)
lines = clean_content.splitlines()

depth = 0
for i, line in enumerate(lines, 1):
    # Find all <div> and </div> in the line in order
    # Regex to find tags
    matches = re.finditer(r'(<div\b|</div>|/>)', line)
    
    for match in matches:
        token = match.group(1)
        if token.startswith('<div'):
            depth += 1
            print(f"L{i}: Open -> Depth {depth}")
        elif token == '</div>':
            depth -= 1
            print(f"L{i}: Close -> Depth {depth}")
            if depth < 0:
                print(f"ERROR: Negative depth at Line {i}!")
                exit(1)
        elif token == '/>': 
             # Self closing div? <div ... /> matches <div\b
             # BUT my regex matched <div\b. does it match <div />?
             # Yes. So I counted it as Open.
             # Now I encounter />?
             # this is complex.
             pass

print(f"Final Depth: {depth}")
if depth != 0:
    print("ERROR: Final depth is not zero")
else:
    print("SUCCESS: Depth is zero")
