"""Integrity checks for the dataset split and effective training data."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parent
rows=json.loads((ROOT/'results/split_manifest.json').read_text())
errors=[];seen={}
for row in rows:
 p=ROOT/row['file'];h=hashlib.sha256(p.read_bytes()).hexdigest()
 if h!=row['sha256']:errors.append('Changed file: '+row['file'])
 if h in seen and seen[h]['split']!=row['split']:errors.append('Cross-split byte duplicate: '+row['file'])
 seen[h]=row
assert not errors,'\n'.join(errors)
print(f'PASS: {len(rows)} files match their manifest; no exact duplicates across splits.')
print('Review results/duplicate_exclusions.json for the two visually reviewed near-duplicate exclusions.')
