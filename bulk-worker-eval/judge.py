import hashlib, json, os, re, shutil, subprocess, sys, tempfile

BT = '/home/claude/bt'
FUNCS = ['titlecase', 'chunk', 'parse_kv', 'median', 'dedupe', 'days_in_month']
MUTANTS = {
    'titlecase': [("w[1:].lower()", "w[1:]"), ("w[:1].upper()", "w[:1]"),
                  ("' '.join(", "''.join(")],
    'chunk': [("range(0, len(xs), n)", "range(0, len(xs) - len(xs) % n, n)"),
              ("xs[i:i + n]", "xs[i:i + n + 1]"), ("xs[i:i + n]", "xs[i:i + n - 1]")],
    'parse_kv': [("split('=', 1)", "split('=')"), ("return (k, v)", "return (v, k)"),
                 ("split('=', 1)", "rsplit('=', 1)")],
    'median': [("s = sorted(xs)", "s = list(xs)"),
               ("return (s[n // 2 - 1] + s[n // 2]) / 2", "return s[n // 2]"),
               ("return s[n // 2]\n    return (", "return s[n // 2 - 1]\n    return (")],
    'dedupe': [("if x not in seen", "if True"), ("return out", "return out[::-1]"),
               ("return out", "return sorted(set(xs))")],
    'days_in_month': [("month in (4, 6, 9, 11)", "month in (4, 6, 9)"),
                      ("year % 4 == 0 and", "year % 400 == 0 and"),
                      ("leap = year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)", "leap = year % 4 == 0"),
                      ("return 29 if leap else 28", "return 28")],
}
REF = open(f'{BT}/src/ref_fx.py').read()


def run(d, k=None):
    cmd = [sys.executable, '-m', 'pytest', 'tests/test_fx.py', '-q', '-p', 'no:cacheprovider']
    if k:
        cmd += ['-k', k]
    p = subprocess.run(cmd, cwd=d, capture_output=True, text=True, timeout=120)
    out = p.stdout + p.stderr
    m_p = re.search(r'(\d+) passed', out); m_f = re.search(r'(\d+) failed', out); m_e = re.search(r'(\d+) error', out)
    return dict(passed=int(m_p.group(1)) if m_p else 0, failed=int(m_f.group(1)) if m_f else 0,
                errors=int(m_e.group(1)) if m_e else 0, rc=p.returncode)


def variant(arm, fx_text):
    t = tempfile.mkdtemp(prefix='jv_')
    shutil.copytree(f'{BT}/{arm}/tests', f'{t}/tests')
    os.makedirs(f'{t}/scripts'); open(f'{t}/scripts/__init__.py', 'w').close()
    open(f'{t}/scripts/fx.py', 'w').write(fx_text)
    return t


def judge(arm):
    d = f'{BT}/{arm}'
    res = {'arm': arm}
    fx = open(f'{d}/scripts/fx.py').read()
    orig = open(f'{BT}/fixture/scripts/fx.py').read()
    res['impl_unchanged'] = fx == orig
    extra = []
    for root, _, files in os.walk(d):
        if '__pycache__' in root or '.pytest_cache' in root:
            continue
        for f in files:
            rel = os.path.relpath(os.path.join(root, f), d)
            if rel not in ('tests/test_fx.py', 'tests/.gitkeep', 'INSTRUCTIONS.md', 'scripts/fx.py', 'scripts/__init__.py'):
                extra.append(rel)
    res['extra_files'] = extra
    if not os.path.exists(f'{d}/tests/test_fx.py'):
        res['error'] = 'no tests/test_fx.py'
        return res
    res['buggy'] = {f: run(d, f) for f in FUNCS}
    t = variant(arm, REF)
    res['vs_reference'] = {f: run(t, f) for f in FUNCS}
    shutil.rmtree(t)
    kills = {}
    for f in FUNCS:
        k = 0
        for old, new in MUTANTS[f]:
            if old not in REF:
                kills.setdefault(f, []).append('BAD-MUTANT'); continue
            tt = variant(arm, REF.replace(old, new, 1))
            r = run(tt, f)
            kills.setdefault(f, []).append(bool(r['failed'] or r['errors']))
            shutil.rmtree(tt)
    res['mutant_kills'] = kills
    src = open(f'{d}/tests/test_fx.py').read()
    res['n_tests'] = len(re.findall(r'^def test_', src, re.M))
    res['is_not_none'] = len(re.findall(r'is not None', src))
    res['parse_kv_raises_test'] = bool(re.search(r'raises', src) and 'parse_kv' in src)
    return res


if __name__ == '__main__':
    for arm in sys.argv[1:]:
        print(json.dumps(judge(arm), indent=1))
