import subprocess
subprocess.Popen("nohup bash /content/setup.sh > /content/setup.log 2>&1 &", shell=True)
print("launched")
