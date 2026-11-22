import subprocess, sys, os
name = os.environ.get('STAGE', 'stage1')
subprocess.Popen(f"nohup bash /content/{name}.sh > /content/{name}.log 2>&1 &", shell=True)
print("launched", name)
