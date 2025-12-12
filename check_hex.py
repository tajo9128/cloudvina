
import os

filepath = 'web/src/pages/3DViewer.jsx'

with open(filepath, 'rb') as f:
    content = f.read()

tail = content[-100:]
print("--- TAIL HEX ---")
print(tail.hex())
print("--- TAIL TEXT ---")
print(tail)
