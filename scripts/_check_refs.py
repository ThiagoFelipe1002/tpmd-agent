from pathlib import Path
import re

docs = list(Path('data/docs').rglob('*.md'))
found = 0
for f in docs:
    t = f.read_text(encoding='utf-8')
    if '## Referências' in t:
        idx = t.index('## Referências')
        print('---', f.name)
        print(repr(t[idx:idx+300]))
        found += 1
        if found >= 3:
            break
count = sum(1 for f in docs if '## Referências' in f.read_text(encoding='utf-8'))
print('Total com Ref:', count)
