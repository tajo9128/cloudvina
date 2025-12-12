
import os

filepath = 'web/src/pages/3DViewer.jsx'

with open(filepath, 'rb') as f:
    content = f.read()

# Replace the specific byte sequence for </div > 
# I'll use regex or string replace.
# String: </div >
fixed_content = content.replace(b'</div >', b'</div>')

# Verify replacement happened
if content == fixed_content:
    print("NO CHANGE - Pattern not found in bytes")
else:
    print("FIXED - Replaced bad bytes")
    with open(filepath, 'wb') as f:
        f.write(fixed_content)
    print("Wrote file back.")
