#!/usr/bin/env python3
"""Every aec.h / aec_internal.h function that takes an Aec handle is exercised by test_handle_ready.c.

usage: check_handle_entries.py include/aec.h test/test_handle_ready.c [include/aec_internal.h]
The entries are the calls between the "BEGIN entries" and "END entries" markers of check_refused(); aec_create() takes no
handle (the NULL of a failed build is checked by the failed-create scenarios of the same file). Exit 1 and a list when a
prototype is missing there or a call there has no prototype."""
import re
import sys

header, test = sys.argv[1], sys.argv[2]
text = ' '.join(open(h).read() for h in [header] + sys.argv[3:])
text = re.sub(r'/\*.*?\*/', ' ', text, flags=re.S)
text = re.sub(r'//[^\n]*', ' ', text)
text = re.sub(r'^\s*#[^\n]*$', ' ', text, flags=re.M)
protos = {}
for m in re.finditer(r'([A-Za-z_][\w \t\*]*?)\b(aec_\w+)\s*\(([^;{}()]*)\)\s*;', text):
    name, params = m.group(2), m.group(3)
    if re.search(r'\bAec\s*\*', params):
        protos[name] = params

src = open(test).read()
i, j = src.index('BEGIN entries'), src.index('END entries')
called = set(re.findall(r'\b(aec_\w+)\s*\(', src[i:j]))
missing = sorted(set(protos) - called)
extra = sorted(called - set(protos))
if missing or extra:
    if missing:
        print('header functions taking an Aec handle that check_refused() does not call: ' + ', '.join(missing))
    if extra:
        print('calls in check_refused() without an Aec-handle prototype in the headers: ' + ', '.join(extra))
    sys.exit(1)
print('check_handle_entries: %d header entry points taking an Aec handle, all exercised on a handle that is not ready' % len(protos))
