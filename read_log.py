
try:
    with open('web/build_err.log', 'rb') as f:
        content = f.read()
    # Try decoding as utf-8, fallback to latin-1
    try:
        text = content.decode('utf-8')
    except UnicodeDecodeError:
        text = content.decode('latin-1')
    
    # Print first 50 lines (where error usually is)
    print("--- HEAD ---")
    print('\n'.join(text.splitlines()[:50]))
    
    # Print last 20 lines (summary)
    print("--- TAIL ---")
    print('\n'.join(text.splitlines()[-20:]))

except Exception as e:
    print(f"Error reading log: {e}")
