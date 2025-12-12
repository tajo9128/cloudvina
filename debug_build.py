
import subprocess
import os

try:
    # Run npm run build in web directory
    # We use shell=True to pick up npm from path
    result = subprocess.run(
        'npm run build',
        cwd='web',
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        encoding='utf-8', 
        errors='replace'
    )
    
    print("EXIT CODE:", result.returncode)
    
    lines = result.stdout.splitlines()
    # Print error summary (look for "error" or "failed")
    found_error = False
    for i, line in enumerate(lines):
        if "error" in line.lower() or "failed" in line.lower():
            print(f"Match at line {i}: {line}")
            # Print context
            for j in range(max(0, i-5), min(len(lines), i+20)):
                print(f"  {lines[j]}")
            found_error = True
            break
            
    if not found_error:
        print("--- TOP 50 LINES ---")
        print('\n'.join(lines[:50]))

except Exception as e:
    print(f"Execution Error: {e}")
