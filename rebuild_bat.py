"""Rebuild install.bat with fresh archive"""
import os

# Read the existing bat file, split at the marker
with open('install/install.bat', 'r', encoding='utf-8') as f:
    content = f.read()

marker = '__SUGIRI_ARCHIVE_MARKER__'
idx = content.rfind(marker)
if idx == -1:
    print("ERROR: marker not found")
    exit(1)

# Everything up to and including the marker line (keep the marker)
before = content[:idx + len(marker)]

# Read new base64 data
with open('install/archive.b64', 'r', encoding='utf-8') as f:
    b64_data = f.read()

# Combine
new_content = before + '\n' + b64_data

# Write
with open('install/install.bat', 'w', encoding='utf-8', newline='\r\n') as f:
    f.write(new_content)

print(f'Done! install.bat updated ({len(new_content)} bytes)')

# Cleanup
os.remove('install/archive.b64')
os.remove('rebuild_archive.py')
print('Cleanup done')
