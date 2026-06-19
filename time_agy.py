import subprocess
import time

t0 = time.time()
res = subprocess.run(['/home/rnela/.local/bin/agy', '--print', 'מה עיר הבירה של צרפת?'], capture_output=True, text=True)
print("Output:", res.stdout.strip())
print("Time:", time.time() - t0)
