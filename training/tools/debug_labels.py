#!/usr/bin/env python3
"""
调试labels.csv文件
"""

import csv
from pathlib import Path

labels_file = Path('data/labels.csv')
labeled_dir = Path('data/labeled')

print(f'Labels file: {labels_file}')
print(f'Labels file exists: {labels_file.exists()}')
print(f'Labeled dir: {labeled_dir}')
print(f'Labeled dir exists: {labeled_dir.exists()}')
print()

with open(labels_file, 'r', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    print(f'CSV Columns: {reader.fieldnames}')
    print()
    
    count = 0
    for row in reader:
        count += 1
        if count <= 5:
            filename = row.get('filename') or row.get('\ufefffilename')
            registration = row.get('registration')
            print(f'Row {count}:')
            print(f'  filename: {filename}')
            print(f'  registration: {registration}')
            
            full_path = labeled_dir / filename
            print(f'  Full path: {full_path}')
            print(f'  Exists: {full_path.exists()}')
            print()
