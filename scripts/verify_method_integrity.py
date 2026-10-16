#!/usr/bin/env python3
"""Check shipped proposal and baseline Python bytes against frozen manifest."""
from pathlib import Path
import hashlib,json,sys
ROOT=Path(__file__).resolve().parents[1]
manifest=json.loads((ROOT/'configs/reviewer/source_integrity.json').read_text())
errors=[]
for item in manifest['files']:
    path=ROOT/item['path']
    if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest()!=item['sha256']:
        errors.append(item['path'])
for section in manifest['scope']:
    expected={x['path'] for x in manifest['files'] if x['path'].startswith(section+'/')}
    actual={str(x.relative_to(ROOT)) for x in (ROOT/section).rglob('*.py')}
    errors+=sorted(actual-expected)
if errors:print('Integrity FAIL:',errors);sys.exit(1)
print('Integrity PASS:',len(manifest['files']),'unchanged proposal/core/baseline files')
