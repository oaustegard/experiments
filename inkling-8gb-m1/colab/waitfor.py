import time, os
name = os.environ.get('STAGE', 'stage1'); marker = name.upper() + '_DONE'
t = time.time(); s = ''
while time.time() - t < float(os.environ.get('MAXWAIT', '3000')):
    p = f'/content/{name}.log'
    s = open(p).read() if os.path.exists(p) else ''
    if marker in s: break
    time.sleep(15)
print(s[-6000:])
