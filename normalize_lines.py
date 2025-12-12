
import os

filepath = 'web/src/pages/3DViewer.jsx'

with open(filepath, 'rb') as f:
    content = f.read()

# Normalize line endings
# Convert all forms of newline to just \n
# Handle \r\r\n (Double CR) first just in case
fixed_content = content.replace(b'\r\r\n', b'\n')
fixed_content = fixed_content.replace(b'\r\n', b'\n')
fixed_content = fixed_content.replace(b'\r', b'\n')

# Deduplicate \n if necessary? No, empty lines are fine.
# But \r\n became \n.
# \r\r\n became \n.
# \r became \n.
# Seems correct.

with open(filepath, 'wb') as f:
    f.write(fixed_content)
print("Normalized line endings to LF.")
