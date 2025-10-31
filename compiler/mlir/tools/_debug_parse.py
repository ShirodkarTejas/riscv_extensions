#!/usr/bin/env python3
import re, sys
p = sys.argv[1]
txt = open(p).read()
for name in ['tile_S','tile_D','tile_M','block_size','global_tokens']:
    m = re.search(rf'{name}\s*=\s*([0-9]+)\s*:\s*i64', txt)
    print(name, '->', m.group(1) if m else 'MISS')
