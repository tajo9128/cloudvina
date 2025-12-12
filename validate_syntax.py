
import re

def check_balance(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    stack = []
    
    # Simple token map
    pairs = {')': '(', '}': '{', ']': '['}
    
    in_comment = False
    
    for line_num, line in enumerate(lines, 1):
        i = 0
        while i < len(line):
            char = line[i]
            
            # Skip comments (naive)
            if not in_comment and line[i:i+2] == '//':
                break
            if not in_comment and line[i:i+2] == '/*':
                in_comment = True
                i += 1
            elif in_comment and line[i:i+2] == '*/':
                in_comment = False
                i += 1
            elif not in_comment:
                if char in '({[':
                    stack.append((char, line_num, i+1))
                elif char in ')}]':
                    if not stack:
                        print(f"Error: Unmatched '{char}' at line {line_num} col {i+1}")
                        return
                    last_open, last_line, last_col = stack.pop()
                    if pairs[char] != last_open:
                        print(f"Error: Mismatched '{char}' at line {line_num} col {i+1}. Expected closing for '{last_open}' from line {last_line}")
                        return
            
            i += 1
            
    if stack:
        first = stack[0]
        print(f"Error: Unclosed '{first[0]}' from line {first[1]} col {first[2]}")
    else:
        print("Success: Braces/Parens Balanced")

check_balance('web/src/pages/3DViewer.jsx')
