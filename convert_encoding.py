
import os

filepath = 'web/src/pages/3DViewer.jsx'

# Try reading as UTF-16
try:
    with open(filepath, 'rb') as f:
        raw = f.read()

    # check for BOM or null bytes
    if b'\x00' in raw:
        print("Detected null bytes (likely UTF-16)")
        try:
            content = raw.decode('utf-16')
        except UnicodeError:
            # Maybe utf-16-le without bom?
            content = raw.decode('utf-16-le')
            
        print("Decoded successfully. Converting to UTF-8...")
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print("File saved as UTF-8.")
    else:
        print("File seems to be UTF-8 already (no null bytes).")
        # Proceed to ensure it is clean anyway
        try:
            content = raw.decode('utf-8')
        except UnicodeError:
            content = raw.decode('latin-1')
            
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print("Re-saved as UTF-8.")
            
except Exception as e:
    print(f"Error: {e}")
