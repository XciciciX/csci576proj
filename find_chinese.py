import re

with open('solve_resize_problem.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

chinese_pattern = re.compile(r'[\u4e00-\u9fff]+')

for i, line in enumerate(lines, 1):
    if chinese_pattern.search(line):
        print(f"{i}: {line.rstrip()}")
