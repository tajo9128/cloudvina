
try:
    with open('committed_file.jsx', 'rb') as f:
        content = f.read()
    
    # Python 3
    text = content.decode('utf-8', errors='replace')
    lines = text.splitlines()
    
    print("--- LAST 20 LINES ---")
    for line in lines[-20:]:
        print(repr(line)) # repr shows hidden chars like \r or spaces

except Exception as e:
    print(f"Error: {e}")
